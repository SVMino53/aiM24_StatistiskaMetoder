import math
import random



def variance(elements: list, population: bool = True) -> float:
    s = 0
    mean = sum(elements)/len(elements)
    for x in elements:
        s += (x - mean)**2
    if population:
        return s/(len(elements))
    else:
        return s/(len(elements) - 1)


elements = []

while True:
    element = input("Add element: ")
    if element == "":
        break
    elif isinstance(element, (float, int)):
        elements.append(float(element))
        print(f"Elements added: {len(elements)}")
    elif element.lower() in ("rand", "random"):
        while True:
            try:
                trys = int(input("How many random elements: "))
                break
            except:
                print("Invalid input!")
        while True:
            try:
                lowest = float(input("Lowest possible value: "))
                break
            except:
                print("Invalid input!")
        while True:
            try:
                highest = float(input("Highest possible value: "))
                break
            except:
                print("Invalid input!")
        elements += [lowest + random.random()*(highest - lowest) for _ in range(trys)]
    else:
        print("Invalid input!")

sorted_elements = sorted(elements)

q1_pos = (len(elements) + 1)/4 - 1
q1_left = int(q1_pos)
q2_pos = (len(elements) + 1)/2 - 1
q2_left = int(q2_pos)
q3_pos = (len(elements) + 1)*3/4 - 1
q3_left = int(q3_pos)

minimum = min(elements)
q1 = sorted_elements[q1_left]*(q1_left + 1 - q1_pos) + sorted_elements[q1_left + 1]*(q1_pos - q1_left)
median = sorted_elements[q2_left]*(q2_left + 1 - q2_pos) + sorted_elements[q2_left + 1]*(q2_pos - q2_left)
q3 = sorted_elements[q3_left]*(q3_left + 1 - q3_pos) + sorted_elements[q3_left + 1]*(q3_pos - q3_left)
maximum = max(elements)
mean = sum(elements)/len(elements)
s2 = variance(elements)
s = math.sqrt(s2)

print()
print(f"Min: {minimum}")
print(f"Q_1: {q1}")
print(f"Median: {median}")
print(f"Q_3: {q3}")
print(f"Max: {maximum}")
print(f"Mean: {mean}")
print(f"Variance: {s2}")
print(f"Standard deviation: {s}")