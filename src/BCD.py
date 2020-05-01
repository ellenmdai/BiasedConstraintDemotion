import numpy as np
from pandas import DataFrame, read_csv
from collections import Counter
from itertools import combinations

class BCD(object):

    """A class that performs Biased Constraint Demotion on given data.

    Attributes:
        vt (DataFrame): violation tableau of data (optional);
        ct (DataFrame): constraint tableau of data;
        markednessConstraints (set): constraint names of all constraints denoted by a 'm:' prefix (or at least not 'f:');
        strata (list of sets): list of each stratum of constraints in order, with more dominant strata coming first;
    """

    def __init__(self, vtPath = None, ctPath = None):
        """
        Parameters:
        ----------
        vtPath : str, optional
            Path to a .csv containing a violation tableau. If both this and ctPath are given, ctPath will be ignored. 
            Default is None.
        ctPath : str, optional
            Path to a .csv containing a constraint tableau. Default is None. If ctPath is None here, then it must be 
            initialized with either loadCt() or generateCtFromVt().
        """
        self.vt = None
        self.ct = None
        self.markednessConstraints = None
        self.strata = []
        if (vtPath):
            self.vt = read_csv(vtPath)
        elif (ctPath):
            self.ct = read_csv(ctPath)
            self.markednessConstraints = set([con for con in self.ct.columns.values[3:] if not con.startswith('f:')])

    def loadVt(self, vtPath):
        self.vt = read_csv(vtPath)

    def loadCt(self, ctPath):
        self.ct = read_csv(ctPath)
        # from constraint names, consider it a markedness constraint if not marked as faithfulness (i.e. does not start with "f:")
        self.markednessConstraints = set([con for con in self.ct.columns.values[3:] if not con.startswith('f:')])

    def generateCtFromVt(self):
        """ Uses self.vt to generate a constraint tableau stored in self.ct, and the set of markedness constraints stored
            in self.markednessConstraints.
        Raises
        ------
        Exception: If for an input, any number other than exactly one row marked as the optimal output.
        """

        vt = self.vt.sort_values(by=['Input', 'Optimal']) # for every input, the optimal output will appear first.
        self.ct = DataFrame(columns=['Input', 'Winner', 'Loser'] + list(vt.columns.values[3:]))
        optimalRow = None
        for i, row in vt.iterrows():
            if (optimalRow is not None and row['Input'] == optimalRow['Input']): # the optimal for this group
                if (not np.isnan(row['Optimal'])):
                    raise Exception("cannot have multiple optimal outputs for singe input:", row['Input'])
                mdpData = optimalRow.values[3:] - row.values[3:]
                mdpData = ['L' if v > 0 else 'W' if v < 0 else '' for v in mdpData]
                self.ct.loc[i] = [row['Input'], optimalRow['Output'], row['Output']] + mdpData
            elif (row['Optimal'] == np.nan):
                raise Exception("must have some optimal output info for:", row['Input'])
            else:
                optimalRow = row
        self.markednessConstraints = set([con for con in self.ct.columns.values[3:] if not con.startswith('f:')])

    def saveCt(self, path):
        self.ct.to_csv(path, index=False)
    
    def saveOrganizedTableau(self, path):
        self.organizeTableau(self.ct, self.strata).to_csv(path, index=False)
    
    def doBCD(self):
        """ Runs the overall BCD algorithm, filling self.strata. """

        ct = self.ct.copy()
        self.strata = []
        faithSets = [] # stack for when determining which set of faithfulness constraints should be ranked
        
        # while we're there are still constraints not yet placed
        while (faithSets or len(ct.columns) > 3): 
            # if we have multiple f-sets to choose from based on the last iteration, continue BCD for each option and 
            # retain the strata of the optimal one.
            if (faithSets):
                bestFaithSet = self.findMinFaithSubset(faithSets)
                self.strata += bestFaithSet[0]
                ct = bestFaithSet[1]
                continue
                    
            iterationResult = self.findNextStratum(ct)
            if (type(iterationResult) == set ): # we got back a single set of constraints to be the next stratum
                self.strata.append(iterationResult)
                # removed resolved data from considerion in future iterations
                self.removeResolvedRowsAndCols(ct, iterationResult)
            else: # we got back a list of possible sets. Store the sets to find which frees up the most m-constraints in the next iteration.
                for faithSet in iterationResult:
                    workingCt = ct.copy()
                    self.removeResolvedRowsAndCols(workingCt, faithSet)
                    faithSets.append(([faithSet], workingCt, 0)) # (potential strata, working ct, # freed markedness constraints)
                    
    def findNextStratum(self, workingCt):
        """ Determines the best set(s) of constraints for the next stratum.
        Parameters:
        ----------
            workingCt (DataFrame): a constraint tableau containing only unranked constraints and unresolved mark-data pairs.
        Returns:
        ----------
            rankNext (list of sets): the potential sets of constraints for the next stratum, either of m-constraints, f-constraits,
                or multiple sets of f-constraints if multiple would free up m-constraints later.
        """

        fuseAll = workingCt.iloc[:, 3:].apply(lambda col: self.fusion(col)) # don't fuse winner, loser, input info
        noL = fuseAll[fuseAll != 'L'] # all constraints that prefer no losers
        noLMarkedness = self.markednessConstraints.intersection(set([constraintName[0] for constraintName in noL.iteritems()]))
        if (noLMarkedness): # if at least of of NoL is a markedness constraint
            rankNext = noLMarkedness
        # otherwise, no markedness constraint can be placed right now, so...
        # if there are still some constraints that prefer losers and if at least one of NoL prefers a winner
        elif (len(noL) < len(fuseAll) and (len(noL[noL == 'W']) > 0)) : 
            freeingFaithSets = self.findPotentialMinFaithSubsets(noL, workingCt)
            if (len(freeingFaithSets) == 1):
                rankNext = freeingFaithSets[0]
            else:
                return freeingFaithSets
        else:
            rankNext = set(noL.index)
        return rankNext
        
    def findPotentialMinFaithSubsets(self, noL, ct):
        """ Finds the smallest sized subsets of f-constraints that frees up an m-constraint for the next iteration.
        Parameters:
        ----------
            noL (Series): Series where the indices are constraints that prefer no losers for the given ct and the values are either
                'W' or 'e'.
            ct: A constraint tableau containing only unranked constraints and unresolved mark-data pairs.
        Returns:
        ----------
            freeingFaithSets (list): All subsets of a certain size that free at least one m-constraint for the next iteraton.
                This may contain just one or multiple sets.
        """

        activeFaith = noL[noL == 'W']
        fSetSize = 0
        freeingFaithSets = []
        # starting with size 1, test all combinations of constraints of that size until at least one combination frees an m-constraint
        while (fSetSize < len(activeFaith) and len(freeingFaithSets) == 0):
            fSetSize += 1
            faithSets = combinations(activeFaith.index, fSetSize)
            # for each possible set, see if placing them as a stratum would free an m-constraint
            for f in faithSets:
                ctFreedomTest = ct.copy()
                self.removeResolvedRowsAndCols(ctFreedomTest, f)
                testFuseAll = ctFreedomTest.iloc[:, 3:].apply(lambda col: self.fusion(col))
                freedMConstraints = self.markednessConstraints.intersection(set( \
                                    [constraintName[0] for constraintName in testFuseAll[testFuseAll != 'L'].iteritems()]))
                # if f frees up a markedness constraint, add to freeingfaithsets
                if (len(freedMConstraints) > 0):
                    freeingFaithSets.append(set(f))
        if (len(freeingFaithSets) == 0):
            # if no such subset exists, return all f-constraints that preferred a winner
            return [set(activeFaith.index)]
        return freeingFaithSets

    def findMinFaithSubset(self, faithSets):
        """ From multiple potential f-constraint subsets, picks out the one that frees the most m-constraints before having
            to place another f-constraint, which is assumed to be the optimal choice. If multiple sets free the same number
            of m-constraints, this returns the first one encountered.
        Parameters:
        ----------
            faithSets (list): A list (treated as a Stack) of tuples in the form (workingStrata, workingCt, numFreedMConstraints) where:
                workingStrata (list): stratified hierachy of constraints, starting with the initial f-set in question, and any other
                    m-constraint sets placed after.
                workingCt (DataFrame): a constraint tableau containing only columns not yet ranked either in self.strata or workingStrata and the
                    mark-data pairs not yet resolved by them.
                numFreedMConstraints: The number of markedness constraints able to be ranked by placing workingStrata's initial f-constraint
                    set.
        Returns:
        ----------
            bestFaithSet (tuple): A tuple in the form (workingStrata, workingCt, numFreedMConstraints) of the f-set that frees the most m-constraints.
        """

        bestFaithSet = ([], None, -1) # any real faithSet will beat this placeholder
        while (faithSets): # while we're still choosing between faithSets
            (workingStrata, workingCt, numFreedMConstraints) = faithSets.pop()
            iterationResult = self.findNextStratum(workingCt)
            # if we get back a single stratum of only markedness constraints, add to numFreedMConstraints and continue iterating
            if (type(iterationResult) == set and self.markednessConstraints.intersection(iterationResult) == iterationResult):
                self.removeResolvedRowsAndCols(workingCt, iterationResult)
                workingStrata.append(iterationResult)
                faithSets.append((workingStrata, workingCt, numFreedMConstraints + len(iterationResult)))
            # if we'd have to place a faithfulness constraint, we've reached the end of our testing for the original faithSet
            # candidate, so update the best faithSet if this one's better
            elif (numFreedMConstraints > bestFaithSet[2]):
                bestFaithSet = (workingStrata, workingCt, numFreedMConstraints)
        return bestFaithSet

    def printStrata(self):
        print("STRATA:")
        for ranking, stratum in enumerate(self.strata, start=1):
            print('Rank', ranking, stratum)

    def calculateRMeasure(self):
        """ Calculates the R-Measure of this particular ranking, which is the sum of the number of m-constraints that dominates
            each f-constraint.
        Returns:
        ----------
            r (int): R-Measure of self.strata
        """

        self.r = 0
        dominatingMCount = 0
        for stratum in self.strata:
            mCount = 0
            for constraint in stratum:
                if constraint in self.markednessConstraints:
                    mCount += 1
                else:
                    self.r += dominatingMCount
            dominatingMCount += mCount
        return self.r

    @staticmethod
    def fusion(constraint): # constraint = a column and its values
        """ logical operation combining  ERCs of multiple data points, as described in RCD the Movie by Alan Prince (2009).
        Parameters:
        ----------
            constraint (series): a column from a constraint tableau where the header is the constraint name and each value comes from 
                a row, where 'W' means the winner is preferred by this constraint and 'L' means the loser is preferred.
        Returns:
        ----------
            'L' if a loser is preferred by any row;
            'e' if neither is prefered by any row;
            'W' if at least one winner and no losers are preferred.
        """

        count = Counter(constraint)
        if 'L' in count:
            return 'L'
        if 'W' not in count:
            return 'e'
        return 'W'

    @staticmethod
    def removeResolvedRowsAndCols(ct, resolvedConstraints):
        """ Gets rid of data resolved by a certain stratum in a constraint tableau. So that it won't be considered in future steps.
        Parameters:
        ----------
            ct (DataFrame): the working constraint tableau.
            resolvedConstraints (set): Names of constraints whose columns and resolved rows are to be eliminated.
        """

        for constraint in resolvedConstraints:
            winnerPreferredIndices = ct[ct[constraint] == 'W'].index
            ct.drop(winnerPreferredIndices, inplace=True)
            ct.drop(constraint, axis=1, inplace=True)

    @staticmethod        
    def organizeTableau(originalCt, strata):
        """ Reorders a constraint tableau to make following the BCD algorithm manually easier. Constraints are sorted left to right
            from most to least dominant, and mark-data pairs are sorted top to bottom by the stratum number that resolves it (i.e.
            rows resolved by more dominant strata are placed towards the top). An additonal 'Rank' column is also added indicating
            which stratum resolves that row.
        Parameters:
        ----------
            originalCt (DataFrame): the constraint tableau that contains all data. This will remain untouched; a copy will be edited.
            strata (list): The stratified hierarchy, where each stratum is a set, whose order will be used to sort the constraint 
            tableau.
        Returns:
        ----------
            ct (DataFrame): A sorted version of originalCt with an additional 'Rank' column.
        """
        ct = originalCt.copy()
        ct['Rank'] = 0
        columnOrder = list(ct.columns)[:3]
        for ranking, stratum in enumerate(strata, start=1):
            columnOrder += list(stratum)
            for constraint in stratum:
                winnerPreferredIndices = ct[(ct[constraint] == 'W') & (ct['Rank'] == 0)].index
                for i in winnerPreferredIndices:
                    ct.loc[i, 'Rank'] = ranking
        ct = ct.sort_values('Rank')
        columnOrder.append('Rank')
        ct = ct.reindex(columns=columnOrder)
        return ct.fillna('')

