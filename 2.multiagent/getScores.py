import subprocess
agents = ['AlphaBetaAgent', 'ExpectimaxAgent', 'MCTS_Agent']
layouts = ['smallClassic', 'mediumClassic']
ghostType = ['RandomGhost', 'DirectionalGhost']
numGames = '10'
totalRuns = len(agents) * len(layouts) * len(ghostType)
i = 1
for layout in layouts:
    for ghost in ghostType:
        for agent in agents:
            command = 'python3 pacman.py -p ' + agent + ' -l ' + layout + ' -g ' + ghost + ' -n '+ numGames +' -q'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            file_object = open('test_results/results' + str(i) + '.txt', 'a')
            title = '\n \n SIMULATION WITH : ' + layout + ' ' + ghost + ' ' + agent + ' \n \n -------------------------------------------------------'
            file_object.write(title)
            file_object.write(str(output))
            file_object.close()
            print('run ' + str(i) + ' of ' + str(totalRuns))
            i+=1