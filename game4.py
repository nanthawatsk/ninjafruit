import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
import math
import json
import os
from pygame import mixer
from typing import Dict, List, Tuple, Optional

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Initialize Pygame and sound
pygame.init()
mixer.init()
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
pygame.display.set_caption("Fruit Ninja with Pose Detection")
clock = pygame.time.Clock()

# Load configuration
CONFIG = {
    "fruit_sizes": {
        "watermelon": [150, 200],
        "pineapple": [200, 250],
        "orange": [100, 100],
        "banana": [100, 150],
        "bomb": [150, 150]
    },
    "game_settings": {
        "initial_lives": 10,
        "initial_time": 180,
        "base_fruit_speed": 5,
        "max_fruit_speed": 15,
        "combo_timeout": 60,
        "stain_duration": 120
    },
    "fruit_points": [1, 2, 3, 4, -5],
    "fruit_weights": [30, 25, 20, 15, 10]
}

# Helper functions
def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> pygame.Surface:
    """Load an image with optional resizing, with fallback to colored surface"""
    try:
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, size) if size else img
    except (pygame.error, FileNotFoundError):
        print(f"Warning: Could not load image {path}")
        surf = pygame.Surface(size if size else (100, 100), pygame.SRCALPHA)
        color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
            255
        )
        surf.fill(color)
        return surf

def load_sound(path: str) -> mixer.Sound:
    """Load a sound file with fallback to dummy sound"""
    try:
        return mixer.Sound(path)
    except (pygame.error, FileNotFoundError):
        print(f"Warning: Could not load sound {path}")
        class DummySound:
            def play(self): pass
        return DummySound()

# Load assets
class Assets:
    def __init__(self):
        self.fruits = {
            "watermelon": {
                "whole": load_image('assets/watermelon.png', CONFIG["fruit_sizes"]["watermelon"]),
                "left": load_image('assets/watermelon_left.png', CONFIG["fruit_sizes"]["watermelon"]),
                "right": load_image('assets/watermelon_right.png', CONFIG["fruit_sizes"]["watermelon"]),
                "stain": load_image('assets/stain_watermelon.png')
            },
            "pineapple": {
                "whole": load_image('assets/pineapple.png', CONFIG["fruit_sizes"]["pineapple"]),
                "left": load_image('assets/pineapple_left.png', CONFIG["fruit_sizes"]["pineapple"]),
                "right": load_image('assets/pineapple_right.png', CONFIG["fruit_sizes"]["pineapple"]),
                "stain": load_image('assets/stain_pineapple.png')
            },
            "orange": {
                "whole": load_image('assets/orange.png', CONFIG["fruit_sizes"]["orange"]),
                "left": load_image('assets/orange_left.png', CONFIG["fruit_sizes"]["orange"]),
                "right": load_image('assets/orange_right.png', CONFIG["fruit_sizes"]["orange"]),
                "stain": load_image('assets/stain_orange.png')
            },
            "banana": {
                "whole": load_image('assets/banana.png', CONFIG["fruit_sizes"]["banana"]),
                "left": load_image('assets/banana_left.png', CONFIG["fruit_sizes"]["banana"]),
                "right": load_image('assets/banana_right.png', CONFIG["fruit_sizes"]["banana"]),
                "stain": load_image('assets/stain_banana.png')
            }
        }
        self.bomb = load_image('assets/bomb.png', CONFIG["fruit_sizes"]["bomb"])
        self.background = load_image('assets/background.jpg', (WIDTH, HEIGHT))
        
        self.sounds = {
            "slice": load_sound('sounds/slice.mp3'),
            "bomb": load_sound('sounds/explosion.mp3'),
            "level_up": load_sound('sounds/level_up.mp3')
        }
        
        self.fonts = {
            "large": pygame.font.SysFont('Arial', 80, bold=True),
            "medium": pygame.font.SysFont('Arial', 50),
            "small": pygame.font.SysFont('Arial', 30)
        }

assets = Assets()

# Game systems
class ComboSystem:
    def __init__(self):
        self.current_combo = 0
        self.highest_combo = 0
        self.combo_timer = 0
        self.combo_timeout = CONFIG["game_settings"]["combo_timeout"]
        
    def add_combo(self) -> int:
        self.current_combo += 1
        self.highest_combo = max(self.highest_combo, self.current_combo)
        self.combo_timer = self.combo_timeout
        return self.current_combo
        
    def reset_combo(self) -> None:
        self.current_combo = 0
        self.combo_timer = 0
        
    def update(self) -> None:
        if self.current_combo > 0:
            self.combo_timer -= 1
            if self.combo_timer <= 0:
                self.reset_combo()
                
    def get_combo_multiplier(self) -> int:
        if self.current_combo < 5:
            return 1
        elif self.current_combo < 10:
            return 2
        elif self.current_combo < 15:
            return 3
        else:
            return 4

class LevelSystem:
    def __init__(self):
        self.current_level = 1
        self.score_to_next_level = 50
        self.fruits_per_level = 5
        self.fruits_sliced = 0
        self.fruit_speed = CONFIG["game_settings"]["base_fruit_speed"]
        self.spawn_rate = 60  # frames between spawns
        
    def check_level_up(self, score: int) -> bool:
        if score >= self.score_to_next_level:
            self.current_level += 1
            self.score_to_next_level += self.current_level * 50
            self.fruit_speed = min(
                CONFIG["game_settings"]["max_fruit_speed"],
                CONFIG["game_settings"]["base_fruit_speed"] + self.current_level
            )
            self.spawn_rate = max(20, 60 - self.current_level * 3)
            assets.sounds["level_up"].play()
            return True
        return False
        
    def get_fruit_type(self) -> int:
        weights = CONFIG["fruit_weights"].copy()
        if self.current_level > 3:
            weights[4] = min(25, 10 + self.current_level)
        return random.choices(range(len(weights)), weights=weights)[0]

class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def add_explosion(self, pos: Tuple[int, int], color: Tuple[int, int, int], count: int = 20) -> None:
        for _ in range(count):
            self.particles.append({
                "pos": list(pos),
                "velocity": [random.uniform(-5, 5), random.uniform(-10, 0)],
                "color": color,
                "size": random.randint(2, 8),
                "life": random.randint(20, 40)
            })
            
    def update(self) -> None:
        for p in self.particles[:]:
            p["pos"][0] += p["velocity"][0]
            p["pos"][1] += p["velocity"][1]
            p["velocity"][1] += 0.2
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
    
    def draw(self, surface: pygame.Surface) -> None:
        for p in self.particles:
            alpha = int(255 * (p["life"] / 40))
            color = (*p["color"], alpha)
            pygame.draw.circle(
                surface, 
                color, 
                (int(p["pos"][0]), int(p["pos"][1])), 
                p["size"]
            )

# Game objects
class Fruit:
    def __init__(self, type_idx: int, pos: Tuple[int, int], level_system: LevelSystem):
        self.type_idx = type_idx
        self.rect = pygame.Rect(pos[0], pos[1], *self.get_size())
        self.sliced = False
        self.angle = 0
        self.rotation_speed = random.choice([-5, -3, -2, 2, 3, 5])
        self.speed = level_system.fruit_speed + random.randint(-1, 1)
        self.rotated_images = {}  # Cache for rotated images
        
    def get_size(self) -> Tuple[int, int]:
        if self.type_idx == 0:  # Watermelon
            return CONFIG["fruit_sizes"]["watermelon"]
        elif self.type_idx == 1:  # Pineapple
            return CONFIG["fruit_sizes"]["pineapple"]
        elif self.type_idx == 2:  # Orange
            return CONFIG["fruit_sizes"]["orange"]
        elif self.type_idx == 3:  # Banana
            return CONFIG["fruit_sizes"]["banana"]
        else:  # Bomb
            return CONFIG["fruit_sizes"]["bomb"]
    
    def get_image(self) -> pygame.Surface:
        if self.type_idx == 0:
            return assets.fruits["watermelon"]["whole"]
        elif self.type_idx == 1:
            return assets.fruits["pineapple"]["whole"]
        elif self.type_idx == 2:
            return assets.fruits["orange"]["whole"]
        elif self.type_idx == 3:
            return assets.fruits["banana"]["whole"]
        else:
            return assets.bomb
    
    def get_rotated(self, angle: int) -> pygame.Surface:
        """Get rotated version of the fruit image with caching"""
        angle = int(angle % 360)
        if angle not in self.rotated_images:
            self.rotated_images[angle] = pygame.transform.rotate(self.get_image(), angle)
        return self.rotated_images[angle]
    
    def update(self, dt: float) -> bool:
        """Update position, return True if fruit is out of screen"""
        self.rect.y += self.speed * dt
        self.angle += self.rotation_speed * dt
        return self.rect.top > HEIGHT
    
    def draw(self, surface: pygame.Surface) -> None:
        if not self.sliced:
            rotated = self.get_rotated(self.angle)
            rect = rotated.get_rect(center=self.rect.center)
            surface.blit(rotated, rect.topleft)

class SlicedFruit:
    def __init__(self, fruit: Fruit, is_left: bool):
        self.type_idx = fruit.type_idx
        self.is_left = is_left
        self.pos = [fruit.rect.x, fruit.rect.y]
        self.velocity = [
            random.randint(-8, -4) if is_left else random.randint(4, 8),
            random.randint(-10, -5)
        ]
        self.timer = 30
        self.angle = 0
        self.rotation_speed = random.choice([-6, -4, -2]) if is_left else random.choice([2, 4, 6])
        
    def get_image(self) -> pygame.Surface:
        fruit_type = ["watermelon", "pineapple", "orange", "banana"][self.type_idx]
        return assets.fruits[fruit_type]["left"] if self.is_left else assets.fruits[fruit_type]["right"]
    
    def update(self, dt: float) -> bool:
        """Update position, return True if particle should be removed"""
        self.pos[0] += self.velocity[0] * dt
        self.pos[1] += self.velocity[1] * dt
        self.velocity[1] += 0.5 * dt
        self.angle += self.rotation_speed * dt
        self.timer -= 1
        return self.timer <= 0
    
    def draw(self, surface: pygame.Surface) -> None:
        rotated = pygame.transform.rotate(self.get_image(), self.angle)
        rect = rotated.get_rect(center=self.pos)
        surface.blit(rotated, rect.topleft)

class Stain:
    def __init__(self, pos: Tuple[int, int], fruit_type: str):
        self.pos = pos
        self.timer = CONFIG["game_settings"]["stain_duration"]
        self.alpha = 255
        self.image = assets.fruits[fruit_type]["stain"]
        
    def update(self) -> bool:
        """Update stain, return True if stain should be removed"""
        self.timer -= 1
        self.alpha = int(255 * (self.timer / CONFIG["game_settings"]["stain_duration"]))
        return self.timer <= 0
    
    def draw(self, surface: pygame.Surface) -> None:
        img = self.image.copy()
        img.set_alpha(self.alpha)
        surface.blit(img, self.pos)

class GameState:
    def __init__(self):
        self.score = 0
        self.lives = CONFIG["game_settings"]["initial_lives"]
        self.time_limit = CONFIG["game_settings"]["initial_time"]
        self.combo_system = ComboSystem()
        self.level_system = LevelSystem()
        self.particle_system = ParticleSystem()
        
        self.fruits: List[Fruit] = []
        self.sliced_fruits: List[SlicedFruit] = []
        self.stains: List[Stain] = []
        
        self.spawn_timer = 0
        self.running = True
        self.game_over = False
        self.paused = False
        
    def reset(self) -> None:
        self.__init__()
        
    def add_fruit(self) -> None:
        fruit_type = self.level_system.get_fruit_type()
        x = random.randint(100, WIDTH - 100)
        self.fruits.append(Fruit(fruit_type, (x, 0), self.level_system))
        
    def slice_fruit(self, fruit_idx: int) -> None:
        fruit = self.fruits[fruit_idx]
        fruit.sliced = True
        assets.sounds["slice"].play()
        
        multiplier = self.combo_system.get_combo_multiplier()
        points = CONFIG["fruit_points"][fruit.type_idx] * multiplier
        self.score += points
        self.level_system.fruits_sliced += 1
        self.combo_system.add_combo()
        
        # Add sliced pieces
        if fruit.type_idx < 4:  # Only for regular fruits
            self.sliced_fruits.append(SlicedFruit(fruit, True))
            self.sliced_fruits.append(SlicedFruit(fruit, False))
            
            # Add particles
            fruit_type = ["watermelon", "pineapple", "orange", "banana"][fruit.type_idx]
            color_map = {
                "watermelon": (255, 0, 0),
                "pineapple": (255, 255, 0),
                "orange": (255, 165, 0),
                "banana": (255, 255, 0)
            }
            self.particle_system.add_explosion(
                fruit.rect.center,
                color_map[fruit_type],
                30
            )
        
        # Replace with new fruit
        self.fruits[fruit_idx] = Fruit(
            self.level_system.get_fruit_type(),
            (random.randint(100, WIDTH - 100), 0),
            self.level_system
        )
        
    def hit_bomb(self, fruit_idx: int) -> None:
        self.fruits[fruit_idx].sliced = True
        assets.sounds["bomb"].play()
        self.lives -= 1
        self.combo_system.reset_combo()
        
        # Add explosion particles
        self.particle_system.add_explosion(
            self.fruits[fruit_idx].rect.center,
            (0, 0, 0),
            50
        )
        
        # Replace with new fruit
        self.fruits[fruit_idx] = Fruit(
            self.level_system.get_fruit_type(),
            (random.randint(100, WIDTH - 100), 0),
            self.level_system
        )

# Game screens
def show_level_intro(screen: pygame.Surface, level: int) -> None:
    screen.fill((0, 0, 0))
    text = assets.fonts["large"].render(f"LEVEL {level}", True, (255, 255, 255))
    screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - 50))
    pygame.display.flip()
    pygame.time.wait(1500)

def show_menu(screen: pygame.Surface) -> str:
    options = ["Start Game", "How to Play", "Quit"]
    selected = 0
    
    while True:
        screen.fill((0, 0, 0))
        
        title = assets.fonts["large"].render("FRUIT NINJA", True, (0, 255, 0))
        screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//4))
        
        for i, option in enumerate(options):
            color = (255, 255, 0) if i == selected else (255, 255, 255)
            text = assets.fonts["medium"].render(option, True, color)
            screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 + i*60))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    return options[selected].lower().replace(" ", "_")
                elif event.key == pygame.K_ESCAPE:
                    return "quit"

def show_how_to_play(screen: pygame.Surface) -> None:
    instructions = [
        "HOW TO PLAY",
        "",
        "Use your hands to slice the fruits!",
        "Avoid the bombs - they cost you lives.",
        "",
        "Press ESC to return to menu"
    ]
    
    while True:
        screen.fill((0, 0, 0))
        
        for i, line in enumerate(instructions):
            if i == 0:  # Title
                text = assets.fonts["large"].render(line, True, (255, 255, 0))
                screen.blit(text, (WIDTH//2 - text.get_width()//2, 50))
            else:
                text = assets.fonts["small"].render(line, True, (255, 255, 255))
                screen.blit(text, (WIDTH//2 - text.get_width()//2, 150 + i*30))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

def show_game_over(screen: pygame.Surface, game_state: GameState) -> None:
    while True:
        screen.fill((0, 0, 0))
        
        game_over = assets.fonts["large"].render("GAME OVER!", True, (255, 0, 0))
        screen.blit(game_over, (WIDTH//2 - game_over.get_width()//2, HEIGHT//2 - 150))
        
        score_text = assets.fonts["medium"].render(f"Final Score: {game_state.score}", True, (255, 255, 255))
        screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 - 50))
        
        combo_text = assets.fonts["medium"].render(f"Highest Combo: {game_state.combo_system.highest_combo}x", True, (0, 255, 0))
        screen.blit(combo_text, (WIDTH//2 - combo_text.get_width()//2, HEIGHT//2))
        
        level_text = assets.fonts["medium"].render(f"Level Reached: {game_state.level_system.current_level}", True, (255, 255, 0))
        screen.blit(level_text, (WIDTH//2 - level_text.get_width()//2, HEIGHT//2 + 50))
        
        restart = assets.fonts["medium"].render("Press ENTER to play again", True, (255, 255, 255))
        screen.blit(restart, (WIDTH//2 - restart.get_width()//2, HEIGHT//2 + 150))
        
        quit_text = assets.fonts["medium"].render("Press ESC to quit", True, (255, 255, 255))
        screen.blit(quit_text, (WIDTH//2 - quit_text.get_width()//2, HEIGHT//2 + 200))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return "restart"
                elif event.key == pygame.K_ESCAPE:
                    return "quit"

# Main game function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Game state
    game_state = GameState()
    
    # Main game loop
    while game_state.running:
        # Handle menu
        menu_choice = show_menu(screen)
        if menu_choice == "quit":
            break
        elif menu_choice == "how_to_play":
            show_how_to_play(screen)
            continue
        
        # Reset game state for new game
        game_state.reset()
        show_level_intro(screen, game_state.level_system.current_level)
        
        # Gameplay loop
        while not game_state.game_over:
            dt = 1
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_state.running = False
                    game_state.game_over = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    game_state.paused = not game_state.paused
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    game_state.game_over = True
            
            if game_state.paused:
                # Show pause screen
                screen.fill((0, 0, 0, 128), pygame.BLEND_RGBA_MULT)
                paused_text = assets.fonts["large"].render("PAUSED", True, (255, 255, 255))
                screen.blit(paused_text, (WIDTH//2 - paused_text.get_width()//2, HEIGHT//2))
                pygame.display.flip()
                continue
            
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with MediaPipe for segmentation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            segmentation_results = selfie_segmentation.process(frame_rgb)
            
            # Create segmentation mask
            condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = pygame.surfarray.array3d(assets.background)
            bg_image = np.transpose(bg_image, (1, 0, 2))  # Convert to (H, W, C)
            bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
            bg_image = cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR)
            bg_image = cv2.flip(bg_image, 1)


            # Apply segmentation - only keep player
            output_image = np.where(condition, frame, bg_image)
            
            # Resize frame to match screen
            output_image = cv2.resize(output_image, (WIDTH, HEIGHT))
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            
            # Draw background first
            screen.blit(assets.background, (0, 0))
            
            # Draw the segmented player
            player_surface = pygame.surfarray.make_surface(np.rot90(output_image))
            screen.blit(player_surface, (0, 0))
            
            # [Rest of your existing game update and rendering code remains unchanged]
            # Update game systems
            game_state.combo_system.update()
            game_state.particle_system.update()
            
            # Check level up
            if game_state.level_system.check_level_up(game_state.score):
                show_level_intro(screen, game_state.level_system.current_level)
            
            # Spawn fruits
            game_state.spawn_timer += 1 * dt
            if game_state.spawn_timer >= game_state.level_system.spawn_rate:
                game_state.add_fruit()
                game_state.spawn_timer = 0
            
            # Update fruits
            for i, fruit in enumerate(game_state.fruits[:]):
                if fruit.update(dt):
                    if fruit.type_idx != 4:  # Not a bomb
                        game_state.lives -= 1
                        game_state.combo_system.reset_combo()
                        
                        # Add stain
                        if fruit.type_idx < 4:  # Only for regular fruits
                            fruit_type = ["watermelon", "pineapple", "orange", "banana"][fruit.type_idx]
                            game_state.stains.append(Stain(
                                (fruit.rect.x, HEIGHT - 30),
                                fruit_type
                            ))
                    
                    # Replace fruit
                    game_state.fruits[i] = Fruit(
                        game_state.level_system.get_fruit_type(),
                        (random.randint(100, WIDTH - 100), 0),
                        game_state.level_system
                    )
                else:
                    fruit.draw(screen)
            
            # Update sliced fruits
            for piece in game_state.sliced_fruits[:]:
                if piece.update(dt):
                    game_state.sliced_fruits.remove(piece)
                else:
                    piece.draw(screen)
            
            # Update stains
            for stain in game_state.stains[:]:
                if stain.update():
                    game_state.stains.remove(stain)
                else:
                    stain.draw(screen)
            
            # Draw particles
            game_state.particle_system.draw(screen)
            
            # Handle hand detection
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                right_x = int((1 - landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x) * WIDTH)
                right_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y * HEIGHT)
                left_x = int((1 - landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x) * WIDTH)
                left_y = int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y * HEIGHT)
                
                # Draw hand indicators
                pygame.draw.circle(screen, (255, 0, 0), (right_x, right_y), 20)
                pygame.draw.circle(screen, (0, 0, 255), (left_x, left_y), 20)
                
                # Check fruit collisions
                for i, fruit in enumerate(game_state.fruits[:]):
                    if (fruit.rect.collidepoint(right_x, right_y) or 
                        fruit.rect.collidepoint(left_x, left_y)):
                        
                        if fruit.type_idx == 4:  # Bomb
                            game_state.hit_bomb(i)
                        else:  # Regular fruit
                            game_state.slice_fruit(i)
            
            # Draw UI
            # Create a semi-transparent background for UI elements
            ui_bg = pygame.Surface((WIDTH, 100), pygame.SRCALPHA)
            ui_bg.fill((0, 0, 0, 128))
            screen.blit(ui_bg, (0, 0))
            
            screen.blit(assets.fonts["medium"].render(f"Score: {game_state.score}", True, (255, 255, 255)), (10, 10))
            screen.blit(assets.fonts["medium"].render(f"Lives: {game_state.lives}", True, (255, 0, 0)), (WIDTH - 200, 10))
            
            level_text = f"Level: {game_state.level_system.current_level} (Next: {game_state.level_system.score_to_next_level})"
            screen.blit(assets.fonts["small"].render(level_text, True, (255, 255, 255)), (WIDTH // 2 - 150, 15))
            
            screen.blit(assets.fonts["medium"].render(f"Time: {int(game_state.time_limit)}", True, (255, 255, 255)), (WIDTH // 2 - 50, 50))
            
            if game_state.combo_system.current_combo > 0:
                combo_color = (0, 255, 0) if game_state.combo_system.current_combo < 10 else \
                            (255, 255, 0) if game_state.combo_system.current_combo < 15 else \
                            (255, 0, 255)
                combo_text = f"Combo: {game_state.combo_system.current_combo}x (x{game_state.combo_system.get_combo_multiplier()})"
                screen.blit(assets.fonts["medium"].render(combo_text, True, combo_color), (WIDTH // 2 - 150, HEIGHT - 50))
                screen.blit(assets.fonts["small"].render(f"Highest: {game_state.combo_system.highest_combo}x", True, (255, 255, 0)), (WIDTH - 300, HEIGHT - 50))
            
            # Check game over conditions
            game_state.time_limit -= 1 / 30 * dt
            if game_state.time_limit <= 0 or game_state.lives <= 0:
                game_state.game_over = True
            
            pygame.display.flip()
            clock.tick(30)
        
        # Game over handling
        if game_state.running:
            result = show_game_over(screen, game_state)
            if result == "quit":
                game_state.running = False
            elif result == "restart":
                game_state.game_over = False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()