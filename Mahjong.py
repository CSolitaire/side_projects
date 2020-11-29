import itertools

def checkChow(hand, i):
	smallest = min(hand[i], hand[i+1], hand[i+2])
	if smallest > 0:
		hand[i:i+3] = [x-smallest for x in hand[i:i+3]]
	return smallest

def checkFourSets(hand):
	found = 0
	for i in range(len(hand)):
		if hand[i] >= 3:
			hand[i] -= 3
			found += 1
		if i+2 < len(hand):
			found += checkChow(hand, i)
	return found == 4

def checkHu(hand):
	for i in range(len(hand)):
		if hand[i] < 2:
			continue
		# identify pair and remove
		remainingHand = list(hand)
		remainingHand[i] -= 2
		if checkFourSets(list(remainingHand)):
			return True
	return False

# generate every possible hand
numEachTile = [list(range(5)) for i in range(9)]
allPossibleHands = list(itertools.product(*numEachTile))

def checkWinningHands():
	count = 0
	winning = 0
	for hand in allPossibleHands:
		if sum(hand) != 14:
			continue
		count += 1
		if checkHu(list(hand)):
			winning += 1
	print (str(winning) + "/" + str(count) + " of all 14-tile hands are winning")
	print(str(100*float(winning)/float(count)) + "%")

def checkGates():
	# keep track of results
	gates = [[] for i in range(10)]
	winningTiles = [{} for i in range(10)]
	for hand in allPossibleHands:
		if sum(hand) != 13:
			continue
		# for each of the 9 possible 14th tile, will it win?
		isWinningTile = [False for i in range(9)] 
		for i in range(9):
			if hand[i] == 4:
				continue
			newHand = list(hand)
			newHand[i] += 1
			if checkHu(list(newHand)):
				isWinningTile[i] = True
		gate = sum(isWinningTile)
		winningTilesIndices = [i+1 for i,j in enumerate(isWinningTile) if j]
		gates[gate].append([hand,winningTilesIndices])
		winningTiles[gate][tuple(winningTilesIndices)] = winningTiles[gate].get(tuple(winningTilesIndices), 0) + 1
	# print out results
	for i in reversed(range(len(gates))):
		print(str(i) + " Gates: " + str(len(gates[i])) + " total hands")
		for gate in gates[i]:
			printHandWinning(gate[0], gate[1])
		print("Number of hands with the same winning tiles:")
		for winning in winningTiles[i]:
			print(str(winning) + ": " + str(winningTiles[i][winning]))

def printHandWinning(hand, winning):
	actualHand = []
	for i in range(9):
		actualHand.extend([i+1]*hand[i])
	output = "Hand: " + str(actualHand) + " Winning tiles: " + str(winning)
	print(output)

print('Mahjong calculator\nSelect one:\n[1] Find all gates\n[2] Calculate number of winning hands')
choice = input('')
if (choice == "1"):
	print('Generating... please wait...')
	checkGates()
elif (choice == "2"):
	print('Calculating... please wait...')
	checkWinningHands()
else:
	print("Sorry, please enter a valid option")