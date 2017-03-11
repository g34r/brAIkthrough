import pygame
from enum import IntEnum
from random import randint

import neat
import os

#init
pygame.font.init()
score_font = pygame.font.SysFont("arial", 30)
score = 0

#screen
screen = pygame.display.set_mode((450, 600))

#player
player = pygame.Rect(155, 500, 70, 15)
player_vel_x = 0.0
player_game_over = False

#ball
ball = pygame.Rect(155, 300, 15, 15)
ball_vel_x = 8.0
ball_vel_y = 8.0

#ball direction thing
class Ball_Direction(IntEnum):
    UP_LEFT = 1,
    UP_RIGHT = 2,
    DOWN_LEFT= 3,
    DOWN_RIGHT = 4,

ball_dir = Ball_Direction(randint(1,2))

def dir_reflect_side():
    global ball_dir
    if ball_dir == Ball_Direction.UP_LEFT:
        ball_dir = Ball_Direction.UP_RIGHT
    elif ball_dir == Ball_Direction.UP_RIGHT:
        ball_dir = Ball_Direction.UP_LEFT
    elif ball_dir == Ball_Direction.DOWN_LEFT:
        ball_dir = Ball_Direction.DOWN_RIGHT
    elif ball_dir == Ball_Direction.DOWN_RIGHT:
        ball_dir = Ball_Direction.DOWN_LEFT

def dir_reflect():
    global ball_dir
    if ball_dir == Ball_Direction.UP_LEFT:
        ball_dir = Ball_Direction.DOWN_LEFT
    elif ball_dir == Ball_Direction.UP_RIGHT:
        ball_dir = Ball_Direction.DOWN_RIGHT
    elif ball_dir == Ball_Direction.DOWN_LEFT:
        ball_dir = Ball_Direction.UP_LEFT
    elif ball_dir == Ball_Direction.DOWN_RIGHT:
        ball_dir = Ball_Direction.UP_RIGHT

#breakable

def breakables_init(breakables):
    for i in range(1, 4):
        breakables.append(pygame.Rect(10, 25*i, 45, 15))
        for j in range(1, 8):
            breakables.append(pygame.Rect(breakables[-1].right+10, breakables[-1].top, 45, 15))

breakables = []
breakables_init(breakables)


#update functions
def player_update():
    global score
    #player move
    player.move_ip(player_vel_x, 0)
    if player.left < 0:
        score -= 10
        player.left = 0
    elif player.right > 450:
        score -= 10
        player.right = 450

def ball_update():
    #ball move
    global breakables
    global ball_dir
    global score
    global player_game_over

    if ball_dir == Ball_Direction.UP_LEFT:
        ball.move_ip(-ball_vel_x, -ball_vel_y)
    elif ball_dir == Ball_Direction.UP_RIGHT:
        ball.move_ip(ball_vel_x, -ball_vel_y)
    elif ball_dir == Ball_Direction.DOWN_LEFT:
        ball.move_ip(-ball_vel_x, ball_vel_y)
    elif ball_dir == Ball_Direction.DOWN_RIGHT:
        ball.move_ip(+ball_vel_x, ball_vel_y)

    #ball refelction
    if ball.left < 0 or ball.right > 450:
        dir_reflect_side()
    elif ball.top < 0:
        dir_reflect()
    elif ball.bottom > 600:

        player_game_over = True

    if ball.colliderect(player):
        dir_reflect()

def breakables_update():
    global score
    if len(breakables) == 0:
        #all cleared
        print("asdf")
    for breakable in breakables:
        if breakable.colliderect(ball):
            breakables.remove(breakable)
            dir_reflect()
            continue

#clocky thing
clock = pygame.time.Clock()
last_key_tick = pygame.time.get_ticks()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, config)

        while True:
            global last_key_tick
            global ball
            global player_vel_x
            global player_game_over
            global ball_dir
            global score
            global breakables

            clock.tick()
            score += 1
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                screen.fill((0, 0, 0))
                pygame.display.flip()
                break
            elif event.type == pygame.KEYDOWN:
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_s]:
                    player.left = 155
                    player_vel_x = 0

                    ball.left = randint(50, 300)
                    ball.top = 300
                    ball_dir = Ball_Direction(randint(1, 2))

                    breakables = []
                    breakables_init(breakables)

                    genome.fitness = score
                    score = 0
                    player_game_over = False
                    break
                # if pressed[pygame.K_RIGHT] and not pressed[pygame.K_LEFT]:
                #     player_vel_x = 1
                # if pressed[pygame.K_LEFT] and not pressed[pygame.K_RIGHT]:
                #     player_vel_x = -1
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT and player_vel_x == 1:
                    player_vel_x = 0
                elif event.key == pygame.K_LEFT and player_vel_x == -1:
                    player_vel_x = 0

            if pygame.time.get_ticks() - last_key_tick >= 5:
                input = (player.left, ball.left)
                output = net.activate(input)[0]

                if output >= 0.5:
                    player_vel_x = 8
                else:
                    player_vel_x = -8

                player_update()
                ball_update()
                breakables_update()

                last_key_tick = pygame.time.get_ticks()

            if player_game_over:
                player.left = 155
                player_vel_x = 0

                ball.left = randint(50, 300)
                ball.top = 300
                ball_dir = Ball_Direction(randint(1,2))

                breakables = []
                breakables_init(breakables)

                genome.fitness = score
                score = 0
                player_game_over = False
                break

            pygame.draw.rect(screen, (255, 255, 255), player)
            pygame.draw.rect(screen, (255, 255, 255), ball)
            screen.blit(score_font.render(str(score), False, (255,255,255)), (0,0))

            for breakable in breakables:
                pygame.draw.rect(screen, (255, 255, 255), breakable)
            pygame.display.flip()
            screen.fill((0, 0, 0))

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    winner = p.run(eval_genomes, 300)

    #node_names = {-4:'player_x', -3: 'player_y', -2:'enemy_x', -1:'enemy_y', 0:"value"}
    #visualize.draw_net(config, winner, True, node_names=node_names)

    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
