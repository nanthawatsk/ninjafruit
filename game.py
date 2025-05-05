import cv2
import mediapipe as mp
import pygame
import random

# Initialize Mediapipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Load background image
background = pygame.transform.scale(pygame.image.load('background.png'), (WIDTH, HEIGHT))

# Load fruit images and resize
fruit_images = [
    pygame.transform.scale(pygame.image.load('assets/watermelon.png'), (40, 40)),
    pygame.transform.scale(pygame.image.load('assets/pineapple.png'), (40, 40)),
    pygame.transform.scale(pygame.image.load('assets/bomb.png'), (40, 40))
]

# Fruit points
fruit_points = [1, 3, 5]

# Colors
WHITE = (255, 255, 255)

# Score and Lives
score = 0
lives = 50
font = pygame.font.SysFont(None, 50)

# Fruit list
fruits = [
    {"rect": pygame.Rect(random.randint(100, WIDTH-100), 0, 40, 40), "type": random.randint(0, 2)} for _ in range(5)
]
fruit_speed = 5

# Main game loop
running = True
cap = cv2.VideoCapture(0)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw background
    screen.blit(background, (0, 0))

    # Move fruits downward
    for fruit in fruits:
        fruit['rect'].y += fruit_speed
        if fruit['rect'].top > HEIGHT:
            fruit['rect'].center = (random.randint(100, WIDTH-100), 0)
            lives -= 1  # Lose a life if fruit falls

        # Draw fruits
        screen.blit(fruit_images[fruit['type']], fruit['rect'])

    # Detect arm positions
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Right hand as knife
        right_wrist_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * WIDTH)
        right_wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * HEIGHT)
        pygame.draw.circle(screen, (255, 0, 0), (right_wrist_x, right_wrist_y), 15)

        # Left hand as second knife
        left_wrist_x = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * WIDTH)
        left_wrist_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * HEIGHT)
        pygame.draw.circle(screen, (0, 0, 255), (left_wrist_x, left_wrist_y), 15)

        # Draw skeleton on game screen
        for connection in mp_pose.POSE_CONNECTIONS:
            start_point = landmarks[connection[0]]
            end_point = landmarks[connection[1]]
            start_pos = (int(start_point.x * WIDTH), int(start_point.y * HEIGHT))
            end_pos = (int(end_point.x * WIDTH), int(end_point.y * HEIGHT))
            pygame.draw.line(screen, (0, 255, 0), start_pos, end_pos, 2)

        # Check collision for both hands
        for fruit in fruits:
            if fruit['rect'].collidepoint(right_wrist_x, right_wrist_y) or fruit['rect'].collidepoint(left_wrist_x, left_wrist_y):
                score += fruit_points[fruit['type']]
                fruit['rect'].center = (random.randint(100, WIDTH-100), 0)
                fruit['type'] = random.randint(0, 2)

    # Draw score and lives
    score_text = font.render(f"Score: {score}", True, (0, 0, 0))
    lives_text = font.render(f"Lives: {lives}", True, (255, 0, 0))
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (WIDTH - 120, 10))

    # Check game over
    if lives <= 0:
        game_over_text = font.render("GAME OVER!", True, (255, 0, 0))
        screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2))
        pygame.display.flip()
        pygame.time.wait(2000)
        running = False

    # Show frame
    cv2.imshow('Webcam Feed', frame)
    pygame.display.flip()
    clock.tick(30)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
