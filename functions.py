import math
import random
# Operative actions
def step_towards_the_baby(xr, yr, perceptions, step_for_the_robot):
    # Calculate next position (x_t+1,y_t+1)
    xn = xr - perceptions[0] * step_for_the_robot * math.sin(math.radians(perceptions[1]))
    yn = yr - perceptions[0] * step_for_the_robot * math.cos(math.radians(perceptions[1]))
    return xn, yn

def perceptions_from_cartesian(xb, yb, xr, yr):
    perceptions = [int(math.sqrt(math.pow(xb - xr, 2) + math.pow(yb - yr, 2))),
                   int(math.degrees(math.atan2(xr - xb, yr - yb)))]
    return perceptions

# Memory functions

def MeasureGoalValue(TimeSilent, MaxTimeToGiveUp):
    reward = MaxTimeToGiveUp - (TimeSilent) / (MaxTimeToGiveUp) * 1
    return reward

def move_randomly():
    xr = random.randrange(80, 900, 1)
    yr = random.randrange(180, 550, 1)
    return xr,yr

def go_to_charge_station():
    xr = 900
    yr = 560
    return xr,yr