"""
This file contains the tasks and instructions for new agent testing.
"""

import json

# Task Instructions
# base_instruction = """Answer this question using simulations: """  # (ALWAYS USE 1/100 cycles and up to 5 molecules for speed.IGNORE the low accuracy!):
base_instruction = ""
parameters = {
    0: {
        "framework": "IRMOF-13",
        "hvf": 0.80,  # from RASPA manual
        "molecule": "carbon dioxide",
        "molecule_l": "n-hexane",
        "rosenbluth": 0.0029442,  # from Aastha
    },
    1: {
        "framework": "MFI_SI",
        "hvf": 0.29,  # from RASPA manual
        "molecule": "nitrogen",
        "molecule_l": "n-pentane",
        "rosenbluth": 0.0197439,  # from Aastha
    },
    2: {
        "framework": "MIL-47",
        "hvf": 0.608,  # from RASPA manual
        "molecule": "ethane",
        "molecule_l": "n-heptane",
        "rosenbluth": 0.0004450,  # from Aastha
    },
}

# Multistep Tasks
ads_dil = "Determine the adsorption enthalpy of {molecule} on {framework} using a simulation at infinite dilution"
ads_dil_l = "Determine the adsorption enthalpy of {molecule_l} on {framework} using a simulation at infinite dilution"
ads_1 = "Determine the adsorption enthalpy of {molecule} on {framework}"
ads_l = "Determine the adsorption enthalpy of {molecule_l} on {framework}"
ads_2 = (
    "Compare the adsorption enthalpies of {molecule} and {molecule_l} on {framework}"
)
h = "Determine the henry coefficient of {molecule} on {framework}"
h_l = "Determine the henry coefficient of {molecule_l} on {framework}"
h_2 = "Determine the henry coefficient of {molecule} and {molecule_l} on {framework}"


tasks_multi = {
    "henry": {"l": h_l, "sl": h_2},
    "ads_dil": {
        "s": ads_dil,
        "l": ads_dil_l,
    },
    "ads_iso": {"s": ads_1, "l": ads_l, "sl": ads_2},
}


# Single Step Tasks
add_hvf = " given the helium void fraction of {hvf}"
add_rb_1 = " given the ideal gas rosenbluth weight of {rosenbluth} for {molecule_l}"

hvf = "Calculate the helium void fraction of {framework}"
surface = "Determine the surface area of {framework}"
rosenbluth_1 = "Calculate the ideal Rosenbluth weights for {molecule_l}"


tasks_single = {
    "hvf": {"x": hvf},
    "surface": {"x": surface},
    "rosenbluth": {"l": rosenbluth_1},
    "henry": {"s": h, "l": h_l + add_rb_1, "sl": h_2 + add_rb_1},
    "ads_dil": {"s": ads_dil + add_hvf, "l": ads_dil_l + add_hvf},
    "ads_iso": {
        "s": ads_1 + add_hvf,
        "l": ads_l + add_hvf + " and" + add_rb_1,
        "sl": ads_2 + add_hvf + " and" + add_rb_1,
    },
}


def task_prompt(task, instruction=base_instruction, parameters=parameters):
    return instruction + task.format(**parameters)


# Render Tasks with parameters

tasks_n = {
    f"{t1}___{t2}": [task_prompt(task, parameters=p) for p in parameters.values()]
    for t1 in tasks_multi.keys()  # outer task
    for t2, task in tasks_multi[t1].items()  # inner task
}

tasks_1 = {
    f"{t1}___{t2}": [task_prompt(task, parameters=p) for p in parameters.values()]
    for t1 in tasks_single.keys()  # outer task
    for t2, task in tasks_single[t1].items()  # inner task
}

if __name__ == "__main__":

    print("Single Molecule Tasks:")
    for x in tasks_1.values():
        for xi in x:
            print(xi)
    print()
    print("\nTwo Molecule Tasks:")
    for x in tasks_n.values():
        for xi in x:
            print(xi)

    json.dump(tasks_1, open("tasks_single.json", "w"), indent=4)
    json.dump(tasks_n, open("tasks_multi.json", "w"), indent=4)
