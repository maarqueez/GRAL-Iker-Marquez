
class World:
    num_of_worlds = 0
    def __init__(self, name):
        self.name = name
        self.number = World.num_of_worlds
        World.num_of_worlds += 1


class Goal:
    num_of_goals = 0
    def __init__(self, name):
        self.name = name
        self.number = Goal.num_of_goals
        Goal.num_of_goals += 1

# Operative policy (not for learning)
class OperativePolicy:
    num_of_operative_policies = 0
    _registry = []
    def __init__(self, name):
        self._registry.append(self)
        self.number = OperativePolicy.num_of_operative_policies
        self.name = name
        OperativePolicy.num_of_operative_policies += 1


class LearningPolicy:
    num_of_learning_policies = 0
    _registry = []
    def __init__(self, name, type, action_file):
        self._registry.append(self)
        self.number = LearningPolicy.num_of_learning_policies
        self.name = name
        self.type = type
        self.action_file = action_file
        LearningPolicy.num_of_learning_policies += 1

class Memory:
    _registry = []
    _next_number = 1

    def __init__(self, world, goal, goalvalue, policy, perceptions, medians):
        self.world = world
        self.goal = goal
        self.goalvalue = goalvalue
        self.policy = policy
        self.perceptions = perceptions
        self.medians = medians
        self.number = Memory._next_number
        Memory._next_number += 1
        Memory._registry.append(self)