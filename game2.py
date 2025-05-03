import cv2
import mediapipe as mp
import pygame
import random
import numpy as np

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Pygame
pygame.init()
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
clock = pygame.time.Clock()

# Load fruit images
fruit_size = 150
whole_watermelon = pygame.transform.scale(pygame.image.load('watermelon.png'), (fruit_size, 200))
whole_pineapple = pygame.transform.scale(pygame.image.load('pineapple.png'), (200, 250))
whole_orange = pygame.transform.scale(pygame.image.load('orange.png'), (100, 100))
bomb_image = pygame.transform.scale(pygame.image.load('bomb.png'), (fruit_size, fruit_size))

# Load sliced images
sliced_watermelon_left = pygame.transform.scale(pygame.image.load('watermelon_left.png'), (fruit_size, 200))
sliced_watermelon_right = pygame.transform.scale(pygame.image.load('watermelon_right.png'), (fruit_size, 200))
sliced_pineapple_left = pygame.transform.scale(pygame.image.load('pineapple_left.png'), (200, 250))
sliced_pineapple_right = pygame.transform.scale(pygame.image.load('pineapple_right.png'), (200, 250))
sliced_orange_left = pygame.transform.scale(pygame.image.load('orange_left.png'), (100, 100))
sliced_orange_right = pygame.transform.scale(pygame.image.load('orange_right.png'), (100, 100))

# Load stain images
stain_watermelon = pygame.image.load('stain_watermelon.png').convert_alpha()
stain_pineapple = pygame.image.load('stain_pineapple.png').convert_alpha()
stain_orange = pygame.image.load('stain_orange.png').convert_alpha()

# Fruit images and points
fruit_images = [whole_watermelon, whole_pineapple, whole_orange, bomb_image]
fruit_points = [1, 2, 3, -5]

# Fonts and game stats
font = pygame.font.SysFont(None, 50)
score = 0
lives = 3
time_limit = 60  # seconds

# Fruits and stains
fruits = []
splitted_fruits = []
stains = []

def create_fruit():
    return {
        "rect": pygame.Rect(random.randint(100, WIDTH - 100), 0, fruit_size, fruit_size),
        "type": random.randint(0, 3),
        "sliced": False,
        "angle": 0,
        "rotation_speed": random.choice([-5, -3, -2, 2, 3, 5])
    }

# Split animation for each fruit
def split_fruit(fruit, left_img, right_img):
    base_x, base_y = fruit['rect'].x, fruit['rect'].y
    return [
        {
            "image": left_img,
            "pos": [base_x, base_y],
            "velocity": [random.randint(-8, -4), random.randint(-10, -5)],
            "timer": 30,
            "angle": 0,
            "rotation_speed": random.choice([-6, -4, -2])
        },
        {
            "image": right_img,
            "pos": [base_x + fruit_size // 2, base_y],
            "velocity": [random.randint(4, 8), random.randint(-10, -5)],
            "timer": 30,
            "angle": 0,
            "rotation_speed": random.choice([2, 4, 6])
        }
    ]

# Main game loop
cap = cv2.VideoCapture(0)
fruits = [create_fruit() for _ in range(5)]
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    # Webcam frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Display webcam background
    frame_resized = cv2.resize(frame, (WIDTH, HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(frame_surface, (0, 0))

    # Update and draw fruits
    for i, fruit in enumerate(fruits):
        fruit['rect'].y += 5
        if fruit['rect'].top > HEIGHT:
            lives -= 1

            stain_img = None
            if fruit['type'] == 0:
                stain_img = stain_watermelon
            elif fruit['type'] == 1:
                stain_img = stain_pineapple
            elif fruit['type'] == 2:
                stain_img = stain_orange

            if stain_img:
                stains.append({
                    "image": stain_img.copy(),
                    "pos": (fruit['rect'].x, HEIGHT - 30),
                    "timer": 120,  # ~4 seconds
                    "alpha": 255
                })

            fruits[i] = create_fruit()
            continue

        if not fruit['sliced']:
            fruit['angle'] += fruit['rotation_speed']
            rotated = pygame.transform.rotate(fruit_images[fruit['type']], fruit['angle'])
            rect = rotated.get_rect(center=fruit['rect'].center)
            screen.blit(rotated, rect.topleft)

    # Update and draw split pieces
    for piece in splitted_fruits[:]:
        piece['pos'][0] += piece['velocity'][0]
        piece['pos'][1] += piece['velocity'][1]
        piece['velocity'][1] += 0.5
        piece['angle'] += piece['rotation_speed']
        rotated = pygame.transform.rotate(piece['image'], piece['angle'])
        rect = rotated.get_rect(center=piece['pos'])
        screen.blit(rotated, rect.topleft)
        piece['timer'] -= 1
        if piece['timer'] <= 0:
            splitted_fruits.remove(piece)

    # Detect hands
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        right_x = int((1 - landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x) * WIDTH)
        right_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y * HEIGHT)
        left_x = int((1 - landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x) * WIDTH)
        left_y = int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y * HEIGHT)

        pygame.draw.circle(screen, (255, 0, 0), (right_x, right_y), 20)
        pygame.draw.circle(screen, (0, 0, 255), (left_x, left_y), 20)

        # Slice fruits
        for i, fruit in enumerate(fruits):
            if fruit['rect'].collidepoint(right_x, right_y) or fruit['rect'].collidepoint(left_x, left_y):
                score += fruit_points[fruit['type']]
                fruit['sliced'] = True
                if fruit['type'] == 0:
                    splitted_fruits.extend(split_fruit(fruit, sliced_watermelon_left, sliced_watermelon_right))
                elif fruit['type'] == 1:
                    splitted_fruits.extend(split_fruit(fruit, sliced_pineapple_left, sliced_pineapple_right))
                elif fruit['type'] == 2:
                    splitted_fruits.extend(split_fruit(fruit, sliced_orange_left, sliced_orange_right))
                elif fruit['type'] == 3:
                    lives -= 1
                fruits[i] = create_fruit()

    # Draw stains
    for stain in stains[:]:
        stain['timer'] -= 1
        if stain['timer'] <= 0:
            stains.remove(stain)
            continue
        stain['alpha'] = int(255 * (stain['timer'] / 120))
        img = stain['image'].copy()
        img.set_alpha(stain['alpha'])
        screen.blit(img, stain['pos'])

    # Draw score and lives
    screen.blit(font.render(f"Score: {score}", True, (0, 0, 0)), (10, 10))
    screen.blit(font.render(f"Lives: {lives}", True, (255, 0, 0)), (WIDTH - 150, 10))
    screen.blit(font.render(f"Time: {int(time_limit)}", True, (0, 0, 0)), (WIDTH // 2 - 50, 10))

    # Game over
    if lives <= 0:
        screen.blit(font.render("GAME OVER!", True, (255, 0, 0)), (WIDTH // 2 - 100, HEIGHT // 2))
        screen.blit(font.render(f"Final Score: {score}", True, (0, 0, 0)), (WIDTH // 2 - 100, HEIGHT // 2 + 50))
        pygame.display.flip()
        pygame.time.wait(2000)
        running = False

    pygame.display.flip()
    clock.tick(30)

    #time limit countdown
    time_limit -= 1 / 30  # Adjust the frame rate
    if time_limit <= 0:
        screen.blit(font.render("TIME'S UP!", True, (255, 0, 0)), (WIDTH // 2 - 100, HEIGHT // 2))
        screen.blit(font.render(f"Final Score: {score}", True, (0, 0, 0)), (WIDTH // 2 - 100, HEIGHT // 2 + 50))
        pygame.display.flip()
        pygame.time.wait(2000)
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
