import math
import time
import random
import pygame
import pickle
import classes
import os
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import threading
import seaborn as sns


def circular_angle_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)


class BabySittingSimulator:
    def __init__(self):
        # Initialize pygame
        pygame.init()

        # Screen setup
        self.screen_width = 962
        self.screen_height = 667
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Iker Marquez Babysitting")

        # Load assets
        self.load_assets()

        # Simulation parameters
        self.margin = 80
        self.max_angle = 180
        self.velocity = 3
        self.step_size = 3
        self.max_crying_time = 20
        self.min_session_time = 60
        self.min_goal_value = 0.85
        self.bad_result_threshold = 0.5
        self.time_to_charge = 1000

        # Initialize entities
        self.init_entities()

        # Load or create memories
        self.memories = self.load_memories()

        # Simulation state
        self.running = True
        self.baby_state = "happy"
        self.start_time = time.time()
        self.cry_start_time = 0
        self.happy_start_time = time.time()
        self.current_policy = None
        self.used_memory_number = None

        # Heatmap display
        self.show_heatmap = False
        self.angle_range = (-180, 180)
        self.distance_range = (0, 500)

    def load_assets(self):
        self.background = pygame.image.load('floor1000.png')
        self.icon = pygame.image.load("umea.jpeg")

        # Baby images
        self.baby_images = {
            "happy": pygame.image.load("baby_happy.jpeg"),
            "crying": pygame.image.load("baby_crying.jpeg")
        }

        # Robot images
        self.robot_images = {
            "default": pygame.image.load("ToyStoryDog.jpeg"),
            "charging": pygame.image.load("charge.jpg")
        }

        # Sounds
        self.sounds = {
            "baby_cry": pygame.mixer.Sound('Baby_Crying.mp3'),
            "baby_laugh": pygame.mixer.Sound('giggling-6799.mp3')
        }

        # Fonts
        self.fonts = {
            "regular": pygame.font.SysFont('Arial', 32)
        }

        # Channels
        self.baby_channel = pygame.mixer.Channel(0)
        self.robot_channel = pygame.mixer.Channel(1)

    def init_entities(self):
        self.baby_pos = (500 - 80, 5)
        self.baby_feet_pos = (500 - 50, 180)
        self.robot_pos = [962 - 95, 667 - 106]

        self.worlds = [classes.World('baby_fond_of_Mozart')]
        self.goals = [classes.Goal('lapse_of_quietness')]

        self.operative_policies = [
            classes.OperativePolicy('move_towards_the_baby'),
            classes.OperativePolicy('go_to_charge_station'),
            classes.OperativePolicy('stop')
        ]

        self.learning_policies = [
            classes.LearningPolicy('dance_in_situ', 'graphical', 'dancing_dog.png'),
            classes.LearningPolicy('move_around_the_baby', 'graphical', 'running_dog.jpg'),
            classes.LearningPolicy('lay_down', 'graphical', 'laydown_dog.png'),
            classes.LearningPolicy('play_lullaby', 'audio', 'rockabyebaby.mp3'),
            classes.LearningPolicy('play_Mozart', 'audio', 'mozart_minuet.mp3'),
            classes.LearningPolicy('bark', 'audio', 'small_dog_barking.mp3')
        ]

    def load_memories(self):
        try:
            with open('CLTM', 'rb') as dbfile:
                memories = pickle.load(dbfile)
                print(f'Memorias cargadas: {len(memories)}')
                return memories
        except:
            print('No se encontraron memorias existentes, comenzando desde cero')
            return []

    def save_memories(self):
        try:
            with open('CLTM', 'wb') as dbfile:
                pickle.dump(self.memories, dbfile)
        except Exception as e:
            print(f"Error guardando memorias: {e}")

    def calculate_perceptions(self):
        dx = self.baby_feet_pos[0] - self.robot_pos[0]
        dy = self.baby_feet_pos[1] - self.robot_pos[1]
        distance = int(math.sqrt(dx ** 2 + dy ** 2))
        angle = int(math.degrees(math.atan2(dy, dx)))
        return [distance, angle]

    def move_randomly(self):
        robot_width = 100
        robot_height = 100
        baby_width = 158
        baby_height = 160

        while True:
            x = random.randrange(self.margin, self.screen_width - self.margin - robot_width)
            y = random.randrange(self.margin, self.screen_height - self.margin - robot_height)
            robot_rect = pygame.Rect(x, y, robot_width, robot_height)
            baby_rect = pygame.Rect(self.baby_pos[0], self.baby_pos[1], baby_width, baby_height)
            if not robot_rect.colliderect(baby_rect):
                break

        self.robot_pos = [x, y]

    def go_to_charge_station(self):
        self.robot_pos = [962 - 95, 667 - 106]
        self.current_policy = "go_to_charge_station"

    def start_baby_crying(self):
        self.baby_state = "crying"
        self.cry_start_time = time.time()
        self.baby_channel.play(self.sounds["baby_cry"], loops=-1)
        self.used_memory_number = None
        self.move_randomly()
        current_perception = self.calculate_perceptions()
        best_memory, best_score = self.find_best_memory(current_perception)

        if best_score > 0.8 and (time.time() - self.cry_start_time) < 5:
            self.current_policy = best_memory.policy
            self.used_memory_number = best_memory.number
            print(f"\n--- Baby crying ---")
            print(f" Acción de MEMORIA #{best_memory.number} (Valor esperado: {best_score:.2f})")
            print(f"  Política: {self.learning_policies[self.current_policy].name}")
        else:
            self.current_policy = random.randint(0, len(self.learning_policies) - 1)
            print(f"\n--- Baby crying ---")
            print(f" Acción ALEATORIA")
            print(f"  Política: {self.learning_policies[self.current_policy].name}")

        policy = self.learning_policies[self.current_policy]
        if policy.type == 'audio':
            self.robot_channel.play(pygame.mixer.Sound(policy.action_file), loops=-1)

    def find_best_memory(self, current_perception):
        best_memory = None
        best_score = 0

        for memory in self.memories:
            dist_diff = 1 - (abs(current_perception[0] - memory.perceptions[0]) / 500)
            angle_difference = circular_angle_diff(current_perception[1], memory.perceptions[1])
            angle_diff = 1 - (angle_difference / 180)
            score = memory.goalvalue * dist_diff * angle_diff
            if score > best_score:
                best_score = score
                best_memory = memory

        return best_memory, best_score

    def stop_baby_crying(self):
        self.baby_state = "happy"
        self.happy_start_time = time.time()
        self.robot_channel.stop()
        self.baby_channel.play(self.sounds["baby_laugh"], loops=-1)
        crying_duration = self.happy_start_time - self.cry_start_time
        goal_value = math.exp(-0.15 * crying_duration)
        print(f"\n--- Baby happy (Time crying: {crying_duration:.1f}s) ---")
        print(f" GoalValue obtenido: {goal_value:.2f}")

        if self.used_memory_number is not None:
            if goal_value < self.bad_result_threshold:
                for i, memory in enumerate(self.memories[:]):
                    if memory.number == self.used_memory_number:
                        deleted_value = memory.goalvalue
                        del self.memories[i]
                        self.save_memories()
                        print(
                            f" Memoria #{self.used_memory_number} BORRADA (Valor obtenido: {goal_value:.2f}, Valor original: {deleted_value:.2f})")
                        break

            if goal_value > self.min_goal_value:
                self.store_new_memory(goal_value)
            else:
                print(f" Valor demasiado bajo ({goal_value:.2f}), NO se guarda")
        else:
            if goal_value > self.min_goal_value:
                self.store_new_memory(goal_value)
            else:
                print(f" Valor demasiado bajo ({goal_value:.2f}), NO se guarda")

    def store_new_memory(self, goal_value):
        new_memory = classes.Memory(
            0, 0, goal_value, self.current_policy,
            self.calculate_perceptions(),
            [60, 15]
        )
        new_memory.number = len(self.memories) + 1
        self.memories.append(new_memory)
        self.save_memories()
        print(f" Memoria #{new_memory.number} GUARDADA (Valor: {goal_value:.2f})")

    def draw(self):
        self.screen.blit(self.background, (0, 0))
        baby_img = self.baby_images[self.baby_state]
        self.screen.blit(baby_img, self.baby_pos)

        if self.current_policy is not None and self.baby_state == "crying":
            policy = self.learning_policies[self.current_policy]
            if policy.type == 'graphical':
                robot_img = pygame.image.load(policy.action_file)
            else:
                robot_img = self.robot_images["default"]
        else:
            robot_img = self.robot_images["charging"] if self.current_policy == "go_to_charge_station" else \
            self.robot_images["default"]

        self.screen.blit(robot_img, self.robot_pos)
        elapsed_time = self.fonts["regular"].render(
            f"Tiempo: {time.time() - self.start_time:.1f}s", True, (255, 255, 255))
        self.screen.blit(elapsed_time, (20, 20))

        if self.show_heatmap:
            self.generate_heatmap()

    def generate_heatmap(self):
        if not self.memories:
            print("No hay memorias para graficar.")
            return

        angles = np.array([m.perceptions[1] for m in self.memories])
        distances = np.array([m.perceptions[0] for m in self.memories])
        values = np.array([m.goalvalue for m in self.memories])

        ANGLE_MIN, ANGLE_MAX = -180, 180
        DIST_MIN, DIST_MAX = 0, 900
        BASE_VALUE = 0.8

        grid_size = 500
        xi = np.linspace(ANGLE_MIN, ANGLE_MAX, grid_size)
        yi = np.linspace(DIST_MIN, DIST_MAX, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        zi = np.full_like(xi, BASE_VALUE)

        for x, y, val in zip(angles, distances, values):
            dist = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)

            radius = 100 + 60 * (val - BASE_VALUE)  # Radio entre 100-250px

            if val > BASE_VALUE:
                influence = (val - BASE_VALUE) * np.exp(-(dist ** 2) / (2 * (radius ** 2)))
                zi = np.maximum(zi, BASE_VALUE + influence)

        plt.figure(figsize=(8, 6))

        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=BASE_VALUE, vmax=1.0)

        img = plt.imshow(zi, extent=[ANGLE_MIN, ANGLE_MAX, DIST_MIN, DIST_MAX],
                         origin='lower', cmap=cmap, aspect='auto',
                         norm=norm, interpolation='bicubic')

        sizes = 30 + 120 * (values - BASE_VALUE) / (1.0 - BASE_VALUE)
        plt.scatter(angles, distances, c=values, cmap=cmap, norm=norm,
                    s=sizes, edgecolor='white', linewidth=0.5, alpha=0.9)

        cbar = plt.colorbar(img, shrink=0.8)
        cbar.set_label('Balioaren Eragina', rotation=270, labelpad=15)
        cbar.set_ticks(np.linspace(BASE_VALUE, 1.0, 5))


        plt.xlabel('Angelua (graduak)', fontsize=12)
        plt.ylabel('Distantzia (pixel)', fontsize=12)
        plt.title('MEMORIEN ERAGIN MAPA', fontsize=14, pad=20)


        plt.grid(True, alpha=0.15, linestyle=':')
        plt.xlim(ANGLE_MIN, ANGLE_MAX)
        plt.ylim(DIST_MIN, DIST_MAX)

        plt.tight_layout()
        plt.show()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_g:
                    self.generate_heatmap()
                elif event.key == pygame.K_r:
                    self.memories = []
                    try:
                        os.remove('CLTM')
                        print("\n--- Todas las memorias borradas ---")
                    except:
                        print("\n--- No había memorias para borrar ---")
                elif event.key == pygame.K_UP and self.baby_state == "happy":
                    self.start_baby_crying()
                elif event.key == pygame.K_DOWN and self.baby_state == "crying":
                    self.stop_baby_crying()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                baby_width = 158
                baby_height = 160
                min_x = baby_width // 2
                max_x = self.screen_width - baby_width // 2
                min_y = baby_height // 2
                max_y = self.screen_height - baby_height // 2
                mouse_x = max(min_x, min(mouse_x, max_x))
                mouse_y = max(min_y, min(mouse_y, max_y))
                self.baby_pos = (mouse_x - baby_width // 2, mouse_y - baby_height // 2)
                self.baby_feet_pos = (mouse_x, mouse_y - 80)
                print(f"\n--- New baby position: ({mouse_x}, {mouse_y}) ---")

    def update(self):
        if self.baby_state == "happy" and (time.time() - self.happy_start_time) > 15:
            self.go_to_charge_station()

    def print_final_report(self):
        print("\n--- Resumen final ---")
        print(f"Tiempo total de simulación: {time.time() - self.start_time:.1f} segundos")
        print(f"Total de memorias almacenadas: {len(self.memories)}")
        for i, memory in enumerate(self.memories):
            print(f"\nMemoria #{i + 1}:")
            print(f"- Política: {self.learning_policies[memory.policy].name}")
            print(f"- GoalValue: {memory.goalvalue:.2f}")
            print(f"- Percepciones: Dist={memory.perceptions[0]}, Ang={memory.perceptions[1]}")

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.update()
        self.print_final_report()
        pygame.quit()


if __name__ == "__main__":
    simulator = BabySittingSimulator()
    simulator.run()