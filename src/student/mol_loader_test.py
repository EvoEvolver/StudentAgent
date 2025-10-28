from student.agent.tools.tools_raspa import MoleculeLoader

if __name__ == '__main__':
    res = MoleculeLoader().run(['methane'])
    print(res)