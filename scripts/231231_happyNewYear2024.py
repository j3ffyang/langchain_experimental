# Python Coding

from colorama import Fore
def heart_shape(msg=" Happy New Year 2024"):
    lines = []
    for y in range(15, -15, -1):
        line = ""
        for x in range(-30, 30):
            f = ((x * 0.05) ** 2 + (y * 0.1) ** 2 - 1) ** 3 - (x * 0.05) ** 2 * (y * 0.1) ** 3
            line += msg[(x - y) % len(msg)] if f <= 0 else " "
        lines.append(line)
    print(Fore.BLUE+"\n".join(lines))
heart_shape()
