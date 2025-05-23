import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import igraph as ig
from collections import Counter
from itertools import combinations 
import sys
epsilon = sys.float_info.epsilon

# triadic-novelty

class CitationData:
    """
    A class for analyzing citation data and computing triadic novelty measures for scholarly publications.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing publication data with columns: 'publicationID', 'references', 'subjects', 'year'.
    baselineRange : tuple, optional
        Tuple (start_year, end_year) for the baseline period. Defaults to (-1, -1), which auto-selects based on data.
    analysisRange : tuple, optional
        Tuple (start_year, end_year) for the analysis period. Defaults to (-1, -1), which auto-selects based on data.
    attractiveness : float, optional
        Attractiveness parameter for base model generation. If None, no model shuffling is performed.
    showProgress : bool, optional
        Whether to show progress bars during computation. Default is True.

    Raises
    ------
    TypeError
        If input data is not a pandas DataFrame or attractiveness is not a number.
    ValueError
        If required columns are missing, year is not numeric, publicationID is not unique, or ranges are invalid.
    """
    def __init__(self, data: pd.DataFrame,
                 baselineRange: tuple = (-1, -1),
                 analysisRange: tuple = (-1, -1),
                 attractiveness: float = None,
                 showProgress: bool = True,):
        self.showProgress = showProgress
        # table with publicationID, references (separated by ;), subjects (separated by ;), year
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        required_columns = ['publicationID', 'references', 'subjects', 'year']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_numeric_dtype(data['year']):
            raise ValueError("Year column must contain numeric values")
        
        # Check for unique publication IDs
        if data['publicationID'].duplicated().any():
            raise ValueError("publicationID column must contain unique values")
        
        
        self.data = data.copy()
        # fill NaN values in references and subjects with empty strings
        self.data['references'] = self.data['references'].fillna('')
        self.data['subjects'] = self.data['subjects'].fillna('')
        
        # index by publicationID
        # make sure publicationID is a string
        self.data['publicationID'] = self.data['publicationID'].astype(str)
        self.data['publicationID_index'] = self.data['publicationID']
        self.data.set_index('publicationID_index', inplace=True)
        
        # if references are not already a list, convert to list of strings (split by ;) skip empty entries
        if(not isinstance(self.data['references'].dropna().iloc[0], list)):
            # split by ; and remove empty entries
            self.data['references'] = self.data['references'].apply(lambda x: [ref.strip() for ref in x.split(';') if ref.strip()])

        # if subjects are not already a list, convert to list of strings (split by ;) skip empty entries
        if(not isinstance(self.data['subjects'].dropna().iloc[0], list)):
            # split by ; and remove empty entries
            self.data['subjects'] = self.data['subjects'].apply(lambda x: [sub.strip() for sub in x.split(';') if sub.strip()])
        # year to int
        self.data['year'] = self.data['year'].astype(int)


        self.year2IDs = self._createYear2IDs()
        self.baselineRange = baselineRange
        
        if baselineRange[0] == -1:
            self.baselineRange = (self.data['year'].min(), baselineRange[1])
        if baselineRange[1] == -1:
            # defaults to 5 years after the minimum year in the data
            if self.data['year'].min() + 5 > self.data['year'].max():
                self.baselineRange = (self.data['year'].min(), self.data['year'].max())
            else:
                self.baselineRange = (self.data['year'].min(), self.data['year'].min() + 5)
        if baselineRange[0] > baselineRange[1]:
            raise ValueError("Baseline range is invalid")
        
        self.analysisRange = self._checkAnalysisRange(analysisRange)

        self.data["referenceSubjects"] = self._subjectsFromReferences()
        self.data['subjects'] = self.data['referenceSubjects']

        if(attractiveness is not None):
            if not isinstance(attractiveness, (int, float)):
                raise TypeError("Attractiveness must be a number")
            if attractiveness <= 0:
                raise ValueError("Attractiveness must be larger than 0")
            self.attractiveness = attractiveness
            self.isModel = True
            self._shuffleSubjects()
        else:
            self.isModel = False
            self.attractiveness = 0

        # if(useReferencesSubjects):
        # self.useReferencesSubjects = useReferencesSubjects
        
        self.subject2IntroductionYear = self._generateSubject2IntroductionYear()
        self.year2IntroducedSubjects = self._generateYear2IntroducedSubjects()
        self.baselineSubjects = self._getBaselineSubjects()
        self.year2TotalReferences = self._createYear2TotalReferences()
        self.referenceSubject2IDs = self._getReferenceSubject2IDs()
        
        
    def _subjectsFromReferences(self):
        # get subjects from references
        # return a new dataframe with publicationID, subjects
        referenceSubjects = []
        for references in self.data['references']:
            entryReferenceSubjects = []
            for reference in references:
                entryReferenceSubjects+=self.data.loc[reference, 'subjects']
            referenceSubjects.append(entryReferenceSubjects)
        return referenceSubjects
    

    def _createYear2IDs(self):
        # create a dictionary with year as key and publicationID as value
        year2IDs = {}
        for year, publicationID in zip(tqdm(self.data['year'], desc="Creating year to publicationID mapping", disable=not self.showProgress), self.data.index):
            if year not in year2IDs:
                year2IDs[year] = []
            year2IDs[year].append(publicationID)
                
        return year2IDs
    
    def _generateSubject2IntroductionYear(self):
        # generate a dictionary with subject as key and introduction year as value
        subject2IntroductionYear = {}
        for year in self.year2IDs:
            for publicationIDs in self.year2IDs[year]:
                for subject in self.data.loc[publicationIDs, 'subjects']:
                    if subject not in subject2IntroductionYear:
                        subject2IntroductionYear[subject] = year
        return subject2IntroductionYear
    
    def _generateYear2IntroducedSubjects(self):
        # generate a dictionary with year as key and introduced subjects as value
        year2IntroducedSubjects = {}
        for subject, introductionYear in self.subject2IntroductionYear.items():
            if introductionYear not in year2IntroducedSubjects:
                year2IntroducedSubjects[introductionYear] = []
            year2IntroducedSubjects[introductionYear].append(subject)
        return year2IntroducedSubjects
    

    def _getBaselineSubjects(self):
        # get the subjects that were introduced in the baseline range
        baselineSubjects = []
        for year in range(self.baselineRange[0], self.baselineRange[1]):
            if year in self.year2IntroducedSubjects:
                baselineSubjects += self.year2IntroducedSubjects[year]
        return set(baselineSubjects)
    


    def _createYear2TotalReferences(self):
        # create a dictionary with year as key and total references as value
        year2TotalReferences = {}
        for year,referencesList in zip(tqdm(self.data['year'], desc="Calculating total references per year", disable=not self.showProgress),self.data['references']):
            if year not in year2TotalReferences:
                year2TotalReferences[year] = 0
            if referencesList==referencesList:
                year2TotalReferences[year] += len(referencesList)
        return year2TotalReferences
    

    def _getReferenceSubject2IDs(self):
        # create a dictionary with subject as key and publicationID as value
        subject2IDs = {}
        for paperID, subjects in zip(tqdm(self.data.index, desc="Processing reference subjects", disable=not self.showProgress), self.data['referenceSubjects']):
            for subject in subjects:
                if subject not in subject2IDs:
                    subject2IDs[subject] = []
                subject2IDs[subject].append(paperID)
        return subject2IDs


    def _checkAnalysisRange(self, analysisRange: tuple):
        if analysisRange[0] == -1:
            analysisRange = (self.baselineRange[1], analysisRange[1])
        if analysisRange[1] == -1:
            analysisRange = (analysisRange[0], self.data['year'].max())
        if analysisRange[0] > analysisRange[1]:
            raise ValueError("Analysis range is invalid")
        if analysisRange[0] < self.baselineRange[1]:
            raise ValueError("Analysis range cannot start before the baseline range ends")
        if analysisRange[1] > self.data['year'].max():
            raise ValueError("Analysis range cannot end after the last year in the data")
        # check if analysisRange is within the data range
        
        return analysisRange

    def calculatePioneerNoveltyScores(self, impactWindowSize: int = 5, returnSubjectLevel: bool = False):
        """
        Calculate pioneer novelty scores for each publication.

        Parameters
        ----------
        impactWindowSize : int, optional
            Number of years after subject introduction to consider for impact calculation. Default is 5.
        returnSubjectLevel : bool, optional
            If True, returns both subject-level and paper-level results. Default is False.

        Returns
        -------
        pd.DataFrame or dict
            DataFrame with columns: 'publicationID', 'introducedSubjects', 'pioneerNoveltyScore',
            'pioneerNoveltyImpact', 'pioneerNoveltyImpactScores'.
            If returnSubjectLevel is True, returns a dict with keys 'subjectPioneer' and 'paperPioneer'.
        """
        # calculate the pioneer novelty score
        # for each subject in the baseline range, get the number of references that were introduced in the baseline range
        # and divide by the total number of references
        # return a dictionary with pioneer novelty score for each paperID and the introduced subject categories
        paperID2PioneerNoveltyScore = {}
        paperID2IntroducedSubjects = {}

        analysisRange = self.analysisRange
        
        coreSubjects = set()
        # prefill the coreSubjects with subjects introduced before the analysis range
        for year in range(self.baselineRange[0], analysisRange[0]):
            if year in self.year2IntroducedSubjects:
                coreSubjects.update(self.year2IntroducedSubjects[year])

        beforeYearSubjects = set(coreSubjects)
        for year in tqdm(range(analysisRange[0], analysisRange[1]), disable=not self.showProgress):
            addedSubjectsOnYear = set()
            for paperID in self.year2IDs[year]:
                subjects = self.data.loc[paperID, 'subjects']
                introducedSubjects = set(subjects) - beforeYearSubjects
                addedSubjectsOnYear.update(introducedSubjects)
                paperID2IntroducedSubjects[paperID] = introducedSubjects
                paperID2PioneerNoveltyScore[paperID] = len(introducedSubjects)
            beforeYearSubjects.update(addedSubjectsOnYear)


        # Pioneer Novelty Impact
        # Subject category Novelty Impact
        subject2NoveltyImpact = {} # PapersReferencing(timewindow,SC1)/TotalPublications(timeWindow)
        for introducingYear,introducedSubjects in self.year2IntroducedSubjects.items():
            for introducedSubject in introducedSubjects:
                if introducingYear < analysisRange[0]:
                    continue
                if introducedSubject not in subject2NoveltyImpact:
                    subject2NoveltyImpact[introducedSubject] = 0
                windowYearRange = range(introducingYear+1,introducingYear+impactWindowSize+1)
                totalPaperCount = 0
                for year in windowYearRange:
                    if year in self.year2IDs:
                        totalPaperCount += len(self.year2IDs[year])
                allPapersReferencing = self.referenceSubject2IDs[introducedSubject]
                windowPapersReferencing = set()
                for paperID in allPapersReferencing:
                    # TODO: Check optimization is needed here
                    if self.data.loc[paperID, 'year'] in windowYearRange: 
                        windowPapersReferencing.add(paperID)
                subject2NoveltyImpact[introducedSubject] = len(windowPapersReferencing) / totalPaperCount if totalPaperCount > 0 else 0

            # PIONEER/PERIPHERY NOVELTY IMPACT of subject categories
            dfSubjectData = pd.DataFrame()
            # add all subject categories, including core
            dfSubjectData["subject"] = list(coreSubjects.union(set(self.subject2IntroductionYear.keys())))
            # Set property Baseline to 1 for core subject categories
            isBaseline = [subject in coreSubjects for subject in dfSubjectData["subject"]]
            dfSubjectData["baseline"] = isBaseline
            # Set property IntroductionYear not all subject categories have an introduction year
            dfSubjectData["introductionYear"] = [self.subject2IntroductionYear[subject] if subject in self.subject2IntroductionYear else np.NaN for subject in dfSubjectData["subject"]]
            # Set property NoveltyImpact
            dfSubjectData["noveltyImpact"] = [subject2NoveltyImpact[subject] if subject in subject2NoveltyImpact else np.NaN for subject in dfSubjectData["subject"]]
            # order by year
            dfSubjectData["introductionYear"] = dfSubjectData["introductionYear"].fillna(-1)
            dfSubjectData.sort_values(by=["introductionYear"],inplace=True,ascending=True)
            # set back nan values
            dfSubjectData["introductionYear"] = dfSubjectData["introductionYear"].replace(-1,np.NaN)
            # reset index
            dfSubjectData.reset_index(drop=True,inplace=True)
            
            
            paperUID2PioneerNoveltyImpact = {}
            paperUID2PioneerNoveltyImpactScores = {}
            for paperUID,introducedSubject in paperID2IntroducedSubjects.items():
                impactScores = []
                for subject in introducedSubject:
                    if subject in subject2NoveltyImpact:
                        impactScores.append(subject2NoveltyImpact[subject])
                if(len(impactScores)):
                    maxImpact = np.nanmax(impactScores)
                else:
                    maxImpact = np.NaN
                paperUID2PioneerNoveltyImpact[paperUID] = maxImpact
                paperUID2PioneerNoveltyImpactScores[paperUID] = impactScores

            
            #             totalPaperCount+=len(year2paperUIDs[year])
            #     allPapersReferencing = subjectCategoryReference2PaperUID[subjectCategory]
            #     windowPapersReferencing = set()
            #     for paperUID in allPapersReferencing:
            #         if paperUID2Year[paperUID] in windowYearRange:
            #             windowPapersReferencing.add(paperUID)
            #     subjectCategory2NoveltyImpact[subjectCategory] = len(windowPapersReferencing)/totalPaperCount
            # subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow] = subjectCategory2NoveltyImpact


        # create dataframe with publicationID, introducedSubjects, pioneerNoveltyScore, pioneerNoveltyImpact
        publicationIDsList = list(self.data.index)
        introducedSubjectsList = [list(paperID2IntroducedSubjects[paperID]) if paperID in paperID2IntroducedSubjects else [] for paperID in publicationIDsList]
        pioneerNoveltyScoresList = [paperID2PioneerNoveltyScore[paperID] if paperID in paperID2PioneerNoveltyScore else np.nan for paperID in publicationIDsList]
        pioneerNoveltyImpactList = [paperUID2PioneerNoveltyImpact[paperID] if paperID in paperUID2PioneerNoveltyImpact else np.nan for paperID in publicationIDsList]
        pioneerNoveltyImpactScoresList = [list(paperUID2PioneerNoveltyImpactScores[paperID]) if paperID in paperUID2PioneerNoveltyImpactScores else [] for paperID in publicationIDsList]
        dfPioneerNoveltyScores = pd.DataFrame({
            'publicationID': publicationIDsList,
            'introducedSubjects': introducedSubjectsList,
            'pioneerNoveltyScore': pioneerNoveltyScoresList,
            'pioneerNoveltyImpact': pioneerNoveltyImpactList,
            'pioneerNoveltyImpactScores': pioneerNoveltyImpactScoresList
        })
        
        if(returnSubjectLevel):
            return {
                'subjectPioneer': dfSubjectData,
                'paperPioneer': dfPioneerNoveltyScores
            }
        else:
            return dfPioneerNoveltyScores


    def calculateMaverickNoveltyScores(self, backwardWindow: int = 5, forwardWindow: int = 5):
        """
        Calculate maverick novelty scores for each publication.

        Parameters
        ----------
        backwardWindow : int, optional
            Number of years before publication to consider for baseline. Default is 5.
        forwardWindow : int, optional
            Number of years after publication to consider for impact. Default is 5.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'publicationID', 'MaverickNoveltyAverage', 'MaverickNoveltySubjectsList',
            'MaverickNoveltyEnhancementAverage', 'MaverickNoveltyDiminishmentAverage'.
        """
        analysisRange = self.analysisRange
        subjectsSet = set([subject for subjects in self.data.loc[:,"subjects"] for subject in subjects])
        if "nan" in subjectsSet:
            subjectsSet.remove("nan")

        index2Subject = {index:subject for index,subject in enumerate(subjectsSet)}
        subject2Index = {subject:index for index,subject in enumerate(subjectsSet)}

        previousNetwork = ig.Graph(len(subject2Index),directed=False)


        scYearlyNetworks = {}
        paper2MaverickNoveltyAverage = {}
        paper2MaverickNoveltyMedian = {}
        paper2MaverickNoveltyMax = {}

        year2MaverickNoveltyAverage = {}
        year2MaverickNoveltyMedian = {}
        year2MaverickNoveltyMax = {}

        paper2MaverickSubjects = {}

        # add edges to the graph based on the SC pairs in the references
        yearRange = range(analysisRange[0], analysisRange[1])
        allYearRange = range(min(self.year2IDs.keys()), max(self.year2IDs.keys())+1)
        for year in tqdm(allYearRange, disable=not self.showProgress):
            newNetwork = previousNetwork.copy()
            previousNetwork = previousNetwork.simplify(combine_edges={"weight":"sum"})
            allYearEdge2NoveltyComponent = {}
            distancesBuffer = {}
            edgesAndWeights = Counter()
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    subjects = self.data.loc[paperUID, 'subjects']
                    if(subjects):
                        subjectIndices = [subject2Index[subject] for subject in subjects]
                        subjectIndices = list(set(subjectIndices))
                        edges = list(combinations(subjectIndices,2))
                        # make edges min(),max()
                        edges = [(min(edge),max(edge)) for edge in edges]
                        edgesSet = set(edges)
                        edgesAndWeights.update(edgesSet)
                        # previous distances between all new edges
                        noveltyComponents = []
                        if(yearRange[0] <= year <= yearRange[-1]):
                            for edge in edgesSet:
                                if(edge in distancesBuffer):
                                    distance = distancesBuffer[edge]
                                else:
                                    distance = previousNetwork.distances(source=edge[0],target=edge[1])[0][0]
                                    distancesBuffer[edge] = distance
                                if(1.0-1.0/distance>epsilon): # check FIX #if(1-1.0/distance>0):
                                    noveltyComponents.append(1-1.0/distance)
                                    allYearEdge2NoveltyComponent[edge] = 1-1.0/distance
                                    if(paperUID not in paper2MaverickSubjects):
                                        paper2MaverickSubjects[paperUID] = []
                                    paper2MaverickSubjects[paperUID].append((edge[0],edge[1],1-1.0/distance))
                            if(noveltyComponents):
                                paper2MaverickNoveltyAverage[paperUID] = np.average(noveltyComponents)
                                paper2MaverickNoveltyMedian[paperUID] = np.median(noveltyComponents)
                                paper2MaverickNoveltyMax[paperUID] = np.max(noveltyComponents)
                            else:
                                paper2MaverickNoveltyAverage[paperUID] = 0
                                paper2MaverickNoveltyMedian[paperUID] = 0
                                paper2MaverickNoveltyMax[paperUID] = 0
                            
            if(yearRange[0] <= year <= yearRange[-1]):
                noveltyYearlyComponents = []
                for edge in edgesAndWeights.keys():
                    if(edge in distancesBuffer):
                        distance = distancesBuffer[edge]
                    else:
                        distance = previousNetwork.distances(source=edge[0],target=edge[1])[0][0]
                        distancesBuffer[edge] = distance
                    if(1.0-1.0/distance>epsilon):
                        noveltyYearlyComponents.append(1-1.0/distance)
                        allYearEdge2NoveltyComponent[edge] = 1-1.0/distance
                if(noveltyYearlyComponents):
                    year2MaverickNoveltyAverage[year] = np.average(noveltyYearlyComponents)
                    year2MaverickNoveltyMedian[year] = np.median(noveltyYearlyComponents)
                    year2MaverickNoveltyMax[year] = np.max(noveltyYearlyComponents)
                else:
                    year2MaverickNoveltyAverage[year] = 0
                    year2MaverickNoveltyMedian[year] = 0
                    year2MaverickNoveltyMax[year] = 0
            
            allYearEdges = list(edgesAndWeights.keys())
            allYearWeights = [edgesAndWeights[edge] for edge in allYearEdges]
            newNetwork.add_edges(allYearEdges,attributes={"weight":allYearWeights})
            previousNetwork = newNetwork
            scYearlyNetworks[year] = newNetwork



        # %%
        # citations received for each pair of subject category over the years
        citationsReceivedSCPairYear = {}
        citationsReceivedSCYear = {}
        citationsReceivedSCPairYearNonNormalized = {}
        citationsReceivedSCYearNonNormalized = {}
        totalPublicationsPerYear = {}
        cumulativeCitationsReceivedSCYear = {}
        allTimePublications = 0

        # add edges to the graph based on the SC pairs in the references
        cummulativeNodesAndWeights = Counter()
        for year in tqdm(yearRange, disable=not self.showProgress):
            edgesAndWeights = Counter()
            nodesAndWeights = Counter()
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    subjects = self.data.loc[paperUID, 'subjects']
                    # subjects = self.data.loc[paperUID, 'referenceSubjects']
                    if(subjects):
                        subjectIndices = [subject2Index[subject] for subject in subjects]
                        subjectIndices = list(set(subjectIndices))
                        edges = list(combinations(subjectIndices,2))
                        # make edges min(),max()
                        edges = [(min(edge),max(edge)) for edge in edges]
                        edgesAndWeights.update(set(edges))
                        nodesAndWeights.update(set(subjectIndices))
            # adjust the weights by the number of publications on that year i.e., / total publications
            totalPublications = len(papers)
            totalPublicationsPerYear[year] = totalPublications
            allTimePublications+=totalPublications
            cummulativeNodesAndWeights.update(nodesAndWeights)
            edgesAndCounts = edgesAndWeights.copy()
            nodesAndCounts = nodesAndWeights.copy()
            if(totalPublications):
                for edge,weight in edgesAndWeights.items():
                    edgesAndWeights[edge] = weight/totalPublications
                for node,weight in nodesAndWeights.items():
                    nodesAndWeights[node] = weight/totalPublications
                
            citationsReceivedSCPairYear[year] = edgesAndWeights
            citationsReceivedSCYear[year] = nodesAndWeights
            
            citationsReceivedSCPairYearNonNormalized[year] = edgesAndCounts
            citationsReceivedSCYearNonNormalized[year] = nodesAndCounts
            

            cumulativeCitationsReceivedSCYear[year] = cummulativeNodesAndWeights.copy()
            for node,weight in cumulativeCitationsReceivedSCYear[year].items():
                cumulativeCitationsReceivedSCYear[year][node] = weight/allTimePublications



        # %%
        #his measure is not good----drop the idea

        # fit a simple line to the data using least squares
        # from scipy.optimize import curve_fit
        # define the true objective function
        # def objective(x, a, b):
        #     return a * x + b

        averageChange = {}
        fitMiddlePointChange = {}
        # hit = 0
        averageChangeRatio = {}
        allPairs = set(list(combinations(subject2Index.values(),2)))
        allconsideredYears = []
        for year in tqdm(yearRange, disable=not self.showProgress):
            for edge in allPairs:
                pastCitations = []
                futureCitations = []
                pastCitationsNonNormalized = []
                futureCitationsNonNormalized = []
                totalPublicationsPast = 0
                totalPublicationsFuture = 0
                for windowYear in range(year-backwardWindow,year):
                    allconsideredYears.append(windowYear)
                    if(windowYear in citationsReceivedSCPairYear):
                        totalPublicationsPast += totalPublicationsPerYear[windowYear]
                    if(windowYear in citationsReceivedSCPairYear):
                        if(edge in citationsReceivedSCPairYear[windowYear]):
                            pastCitations.append(citationsReceivedSCPairYear[windowYear][edge])
                            pastCitationsNonNormalized.append(citationsReceivedSCPairYearNonNormalized[windowYear][edge])
                        else:
                            pastCitations.append(0)
                            pastCitationsNonNormalized.append(0)
                    else:
                        pastCitations.append(0)
                        pastCitationsNonNormalized.append(0)
                for windowYear in range(year+1,year+forwardWindow+1):
                    if(windowYear in citationsReceivedSCPairYear):
                        totalPublicationsFuture += totalPublicationsPerYear[windowYear]
                    if(windowYear in citationsReceivedSCPairYear):
                        if(edge in citationsReceivedSCPairYear[windowYear]):
                            futureCitations.append(citationsReceivedSCPairYear[windowYear][edge])
                            futureCitationsNonNormalized.append(citationsReceivedSCPairYearNonNormalized[windowYear][edge])
                        else:
                            futureCitations.append(0)
                            futureCitationsNonNormalized.append(0)
                    else:
                        futureCitations.append(0) 
                        futureCitationsNonNormalized.append(0)
                
                if(year not in averageChange):
                    averageChange[year] = {}
                    averageChangeRatio[year] = {}
                    fitMiddlePointChange[year] = {}
                if(futureCitations and pastCitations):
                    averageChange[year][edge] = np.average(futureCitations)-np.average(pastCitations)
                    pastLinearFit = np.polyfit(range(len(pastCitations)),pastCitations,1)
                    futureLinearFit = np.polyfit(range(len(futureCitations)),futureCitations,1)
                    pastMidPoint = pastLinearFit[0]*len(pastCitations)/2+pastLinearFit[1]
                    if(totalPublicationsFuture and totalPublicationsPast):
                        futureMidPoint = futureLinearFit[0]*len(futureCitations)/2+futureLinearFit[1]
                        fitMiddlePointChange[year][edge] = futureMidPoint/totalPublicationsFuture-pastMidPoint/totalPublicationsPast
                    
                if(np.average(pastCitations)):
                    averageChangeRatio[year][edge] = np.average(futureCitations)/np.average(pastCitations)



        # get the new neighbors from the  BASED ON THE CHANGE OF THE PROPORTION CHANGE RATE
        # timeWindow

        MaverickNoveltyEnhancedAverageByYear = {}
        MaverickNoveltyDiminishAverageByYear = {}
        MaverickNoveltyImpactAverageByYear = {}

        MaverickNoveltyEnhancedAverageByPaperID = {}
        MaverickNoveltyDiminishAverageByPaperID = {}
        MaverickNoveltyImpactAverageByPaperID = {}

        MaverickNoveltyEnhancedMinByYear = {}
        MaverickNoveltyDiminishMinByYear = {}
        MaverickNoveltyImpactMinByYear = {}

        MaverickNoveltyEnhancedMinByPaperID = {}
        MaverickNoveltyDiminishMinByPaperID = {}
        MaverickNoveltyImpactMinByPaperID = {}

        MaverickNoveltyEnhancedMaxByYear = {}
        MaverickNoveltyDiminishMaxByYear = {}
        MaverickNoveltyImpactMaxByYear = {}

        MaverickNoveltyEnhancedMaxByPaperID = {}
        MaverickNoveltyDiminishMaxByPaperID = {}
        MaverickNoveltyImpactMaxByPaperID = {}

        MaverickNoveltyEnhancedMedianByYear = {}
        MaverickNoveltyDiminishMedianByYear = {}
        MaverickNoveltyImpactMedianByYear = {}

        MaverickNoveltyEnhancedMedianByPaperID = {}
        MaverickNoveltyDiminishMedianByPaperID = {}
        MaverickNoveltyImpactMedianByPaperID = {}

        with tqdm(total=len(self.data), desc="Calculating Maverick Novelty Scores", disable=not self.showProgress) as progressBar:
            accumulatedEdgesYearBefore = set()
            for year in yearRange:
                edgesAndWeights = Counter()
                previousYearNetwork = scYearlyNetworks[year]
                if(year in self.year2IDs):
                    papers = self.year2IDs[year]
                    for paperUID in papers:
                        progressBar.update(1)
                        subjects = self.data.loc[paperUID, 'subjects']
                        if subjects:
                            subjectIndices = [subject2Index[subject] for subject in subjects]
                            subjectIndices = list(set(subjectIndices))
                            edges = list(combinations(subjectIndices,2))
                            # make edges min(),max()
                            edges = set([(min(edge),max(edge)) for edge in edges])
                            # keep only edges that were not in the previous years network (yearNetwork)
                            edges = edges - accumulatedEdgesYearBefore
                            edgesAndWeights.update(edges)

                            neighborEdges = set()
                            for edge in edges:
                                neighborsPair = previousYearNetwork.neighborhood(vertices=edge,order=1)
                                for neighbors in neighborsPair:
                                    for neighbor in neighbors:
                                        if(neighbor!=edge[0] and neighbor!=edge[1]):
                                            neighborEdges.add((min(edge[0],neighbor),max(edge[0],neighbor)))
                                            neighborEdges.add((min(edge[1],neighbor),max(edge[1],neighbor)))
                            neighborEdges = neighborEdges - set(edges)
                            differences = [averageChange[year][edge] if edge in averageChange[year] else 0  for edge in neighborEdges]
                            if(differences):
                                MaverickNoveltyImpactAverageByPaperID[paperUID] = np.average(differences)
                                MaverickNoveltyImpactMinByPaperID[paperUID] = np.min(differences)
                                MaverickNoveltyImpactMaxByPaperID[paperUID] = np.max(differences)
                                MaverickNoveltyImpactMedianByPaperID[paperUID] = np.median(differences)
                            else:
                                MaverickNoveltyImpactAverageByPaperID[paperUID] = 0
                                MaverickNoveltyImpactMinByPaperID[paperUID] = 0
                                MaverickNoveltyImpactMaxByPaperID[paperUID] = 0
                                MaverickNoveltyImpactMedianByPaperID[paperUID] = 0
                            
                            # enhanced novelty for positives
                            positiveDifferences = [value for value in differences if value>0] # Check if this is correct
                            if(positiveDifferences):
                                MaverickNoveltyEnhancedAverageByPaperID[paperUID] = np.average(positiveDifferences)
                                MaverickNoveltyEnhancedMinByPaperID[paperUID] = np.min(positiveDifferences)
                                MaverickNoveltyEnhancedMaxByPaperID[paperUID] = np.max(positiveDifferences)
                                MaverickNoveltyEnhancedMedianByPaperID[paperUID] = np.median(positiveDifferences)
                            else:
                                MaverickNoveltyEnhancedAverageByPaperID[paperUID] = 0
                                MaverickNoveltyEnhancedMinByPaperID[paperUID] = 0
                                MaverickNoveltyEnhancedMaxByPaperID[paperUID] = 0
                                MaverickNoveltyEnhancedMedianByPaperID[paperUID] = 0
                            
                            # diminished novelty for negatives
                            negativeDifferences = [value for value in differences if value<0]
                            if(negativeDifferences):
                                MaverickNoveltyDiminishAverageByPaperID[paperUID] = np.average(negativeDifferences)
                                MaverickNoveltyDiminishMinByPaperID[paperUID] = np.min(negativeDifferences)
                                MaverickNoveltyDiminishMaxByPaperID[paperUID] = np.max(negativeDifferences)
                                MaverickNoveltyDiminishMedianByPaperID[paperUID] = np.median(negativeDifferences)
                            else:
                                MaverickNoveltyDiminishAverageByPaperID[paperUID] = 0
                                MaverickNoveltyDiminishMinByPaperID[paperUID] = 0
                                MaverickNoveltyDiminishMaxByPaperID[paperUID] = 0
                                MaverickNoveltyDiminishMedianByPaperID[paperUID] = 0

                    # get neighbor edges to the source and targets of the edges from the network
                    neighborEdges = set()
                    for edge in edgesAndWeights.keys():
                        neighborsPair = previousYearNetwork.neighborhood(vertices=edge,order=1)
                        for neighbors in neighborsPair:
                            for neighbor in neighbors:
                                if(neighbor!=edge[0] and neighbor!=edge[1]):
                                    neighborEdges.add((min(edge[0],neighbor),max(edge[0],neighbor)))
                                    neighborEdges.add((min(edge[1],neighbor),max(edge[1],neighbor)))
                    # remove original edges
                    neighborEdges = neighborEdges - set(edgesAndWeights)
                    differences = [averageChange[year][edge] if edge in averageChange[year] else 0  for edge in neighborEdges]
                    if(differences):
                        MaverickNoveltyImpactAverageByYear[year] = np.average(differences)
                        MaverickNoveltyImpactMinByYear[year] = np.min(differences)
                        MaverickNoveltyImpactMaxByYear[year] = np.max(differences)
                        MaverickNoveltyImpactMedianByYear[year] = np.median(differences)
                    else:
                        MaverickNoveltyImpactAverageByYear[year] = 0
                        MaverickNoveltyImpactMinByYear[year] = 0
                        MaverickNoveltyImpactMaxByYear[year] = 0
                        MaverickNoveltyImpactMedianByYear[year] = 0

                    # enhanced novelty for positives
                    positiveDifferences = [value for value in differences if value>=0]
                    if(positiveDifferences):
                        MaverickNoveltyEnhancedAverageByYear[year] = np.average(positiveDifferences)
                        MaverickNoveltyEnhancedMinByYear[year] = np.min(positiveDifferences)
                        MaverickNoveltyEnhancedMaxByYear[year] = np.max(positiveDifferences)
                        MaverickNoveltyEnhancedMedianByYear[year] = np.median(positiveDifferences)
                    else:
                        MaverickNoveltyEnhancedAverageByYear[year] = 0
                        MaverickNoveltyEnhancedMinByYear[year] = 0
                        MaverickNoveltyEnhancedMaxByYear[year] = 0
                        MaverickNoveltyEnhancedMedianByYear[year] = 0
                    
                    # diminished novelty for negatives
                    negativeDifferences = [value for value in differences if value<0]
                    if(negativeDifferences):
                        MaverickNoveltyDiminishAverageByYear[year] = np.average(negativeDifferences)
                        MaverickNoveltyDiminishMinByYear[year] = np.min(negativeDifferences)
                        MaverickNoveltyDiminishMaxByYear[year] = np.max(negativeDifferences)
                        MaverickNoveltyDiminishMedianByYear[year] = np.median(negativeDifferences)
                    else:
                        MaverickNoveltyDiminishAverageByYear[year] = 0
                        MaverickNoveltyDiminishMinByYear[year] = 0
                        MaverickNoveltyDiminishMaxByYear[year] = 0
                        MaverickNoveltyDiminishMedianByYear[year] = 0
                else:
                    MaverickNoveltyEnhancedAverageByYear[year] = 0
                    MaverickNoveltyDiminishAverageByYear[year] = 0
                    MaverickNoveltyEnhancedMinByYear[year] = 0
                    MaverickNoveltyDiminishMinByYear[year] = 0
                    MaverickNoveltyEnhancedMaxByYear[year] = 0
                    MaverickNoveltyDiminishMaxByYear[year] = 0
                    MaverickNoveltyEnhancedMedianByYear[year] = 0
                    MaverickNoveltyDiminishMedianByYear[year] = 0
                    MaverickNoveltyImpactAverageByYear[year] = 0
                    MaverickNoveltyImpactMinByYear[year] = 0
                    MaverickNoveltyImpactMaxByYear[year] = 0
                    MaverickNoveltyImpactMedianByYear[year] = 0
                    # MaverickNoveltyByYear
                accumulatedEdgesYearBefore = accumulatedEdgesYearBefore.union(edgesAndWeights.keys())
        # create tables for paperID
        publicationIDsList = list(self.data.index)
        #         paper2MaverickNoveltyAverage = {}
        # paper2MaverickNoveltyMedian = {}
        # paper2MaverickNoveltyMax = {}

        # year2MaverickNoveltyAverage = {}
        # year2MaverickNoveltyMedian = {}
        # year2MaverickNoveltyMax = {}
        tableData = {}
        tableData["publicationID"] = publicationIDsList
        tableData["MaverickNoveltyAverage"] = [paper2MaverickNoveltyAverage[paperID] if paperID in paper2MaverickNoveltyAverage else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyMedianList"] = [paper2MaverickNoveltyMedian[paperID] if paperID in paper2MaverickNoveltyMedian else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyMaxList"] = [paper2MaverickNoveltyMax[paperID] if paperID in paper2MaverickNoveltyMax else np.nan for paperID in publicationIDsList]
        tableData["MaverickNoveltySubjectsList"] = [list(paper2MaverickSubjects[paperID]) if paperID in paper2MaverickSubjects else [] for paperID in publicationIDsList]

        # impacts
        tableData["MaverickNoveltyEnhancementAverage"] = [MaverickNoveltyEnhancedAverageByPaperID[paperID] if paperID in MaverickNoveltyEnhancedAverageByPaperID else np.nan for paperID in publicationIDsList]
        tableData["MaverickNoveltyDiminishmentAverage"] = [MaverickNoveltyDiminishAverageByPaperID[paperID] if paperID in MaverickNoveltyDiminishAverageByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyImpactAverage"] = [MaverickNoveltyImpactAverageByPaperID[paperID] if paperID in MaverickNoveltyImpactAverageByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyEnhancedMin"] = [MaverickNoveltyEnhancedMinByPaperID[paperID] if paperID in MaverickNoveltyEnhancedMinByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyDiminishMin"] = [MaverickNoveltyDiminishMinByPaperID[paperID] if paperID in MaverickNoveltyDiminishMinByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyImpactMin"] = [MaverickNoveltyImpactMinByPaperID[paperID] if paperID in MaverickNoveltyImpactMinByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyEnhancedMax"] = [MaverickNoveltyEnhancedMaxByPaperID[paperID] if paperID in MaverickNoveltyEnhancedMaxByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyDiminishMax"] = [MaverickNoveltyDiminishMaxByPaperID[paperID] if paperID in MaverickNoveltyDiminishMaxByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyImpactMax"] = [MaverickNoveltyImpactMaxByPaperID[paperID] if paperID in MaverickNoveltyImpactMaxByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyEnhancedMedian"] = [MaverickNoveltyEnhancedMedianByPaperID[paperID] if paperID in MaverickNoveltyEnhancedMedianByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyDiminishMedian"] = [MaverickNoveltyDiminishMedianByPaperID[paperID] if paperID in MaverickNoveltyDiminishMedianByPaperID else np.nan for paperID in publicationIDsList]
        # tableData["MaverickNoveltyImpactMedian"] = [MaverickNoveltyImpactMedianByPaperID[paperID] if paperID in MaverickNoveltyImpactMedianByPaperID else np.nan for paperID in publicationIDsList]
        dfMaverickNoveltyScores = pd.DataFrame(tableData)

        return dfMaverickNoveltyScores


    def calculateVanguardNoveltyScores(self, weightsCount: int = 4):
        """
        Calculate vanguard novelty scores for each publication.

        Parameters
        ----------
        weightsCount : int, optional
            Number of bins for edge weights in novelty calculation. Default is 4.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'publicationID', 'VanguardNovelty', 'VanguardNoveltyRank', 'VanguardNoveltyImpact'.
        """
        analysisRange = self.analysisRange
        subjectsSet = set([subject for subjects in self.data.loc[:,"subjects"] for subject in subjects])
        if "nan" in subjectsSet:
            subjectsSet.remove("nan")

        index2Subject = {index:subject for index,subject in enumerate(subjectsSet)}
        subject2Index = {subject:index for index,subject in enumerate(subjectsSet)}

        previousNetwork = ig.Graph(len(subject2Index),directed=False)


        scYearlyNetworks = {}
        # add edges to the graph based on the SC pairs in the references
        yearRange = range(analysisRange[0], analysisRange[1])
        allYearRange = range(min(self.year2IDs.keys()), max(self.year2IDs.keys())+1)
        for year in tqdm(allYearRange, disable=not self.showProgress):
            newNetwork = previousNetwork.copy()
            previousNetwork = previousNetwork.simplify(combine_edges={"weight":"sum"})
            distancesBuffer = {}
            edgesAndWeights = Counter()
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    subjects = self.data.loc[paperUID, 'subjects']
                    if(subjects):
                        subjectIndices = [subject2Index[subject] for subject in subjects]
                        subjectIndices = list(set(subjectIndices))
                        edges = list(combinations(subjectIndices,2))
                        # make edges min(),max()
                        edges = [(min(edge),max(edge)) for edge in edges]
                        edgesSet = set(edges)
                        edgesAndWeights.update(edgesSet)
                        # previous distances between all new edges
            allYearEdges = list(edgesAndWeights.keys())
            allYearWeights = [edgesAndWeights[edge] for edge in allYearEdges]
            newNetwork.add_edges(allYearEdges,attributes={"weight":allYearWeights})
            previousNetwork = newNetwork
            scYearlyNetworks[year] = newNetwork


        # %%
        # citations received for each pair of subject category over the years
        citationsReceivedSCPairYear = {}
        citationsReceivedSCYear = {}
        citationsReceivedSCPairYearNonNormalized = {}
        citationsReceivedSCYearNonNormalized = {}
        totalPublicationsPerYear = {}
        cumulativeCitationsReceivedSCYear = {}
        allTimePublications = 0

        # add edges to the graph based on the SC pairs in the references
        cummulativeNodesAndWeights = Counter()
        for year in tqdm(yearRange, disable=not self.showProgress):
            edgesAndWeights = Counter()
            nodesAndWeights = Counter()
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    subjects = self.data.loc[paperUID, 'subjects']
                    # subjects = self.data.loc[paperUID, 'referenceSubjects']
                    if(subjects):
                        subjectIndices = [subject2Index[subject] for subject in subjects]
                        subjectIndices = list(set(subjectIndices))
                        edges = list(combinations(subjectIndices,2))
                        # make edges min(),max()
                        edges = [(min(edge),max(edge)) for edge in edges]
                        edgesAndWeights.update(set(edges))
                        nodesAndWeights.update(set(subjectIndices))
            # adjust the weights by the number of publications on that year i.e., / total publications
            totalPublications = len(papers)
            totalPublicationsPerYear[year] = totalPublications
            allTimePublications+=totalPublications
            cummulativeNodesAndWeights.update(nodesAndWeights)
            edgesAndCounts = edgesAndWeights.copy()
            nodesAndCounts = nodesAndWeights.copy()
            if(totalPublications):
                for edge,weight in edgesAndWeights.items():
                    edgesAndWeights[edge] = weight/totalPublications
                for node,weight in nodesAndWeights.items():
                    nodesAndWeights[node] = weight/totalPublications
                
            citationsReceivedSCPairYear[year] = edgesAndWeights
            citationsReceivedSCYear[year] = nodesAndWeights
            
            citationsReceivedSCPairYearNonNormalized[year] = edgesAndCounts
            citationsReceivedSCYearNonNormalized[year] = nodesAndCounts
            

            cumulativeCitationsReceivedSCYear[year] = cummulativeNodesAndWeights.copy()
            for node,weight in cumulativeCitationsReceivedSCYear[year].items():
                cumulativeCitationsReceivedSCYear[year][node] = weight/allTimePublications



        VanguardNoveltyByPaperID = {}
        VanguardNoveltyByYear = {}

        totalNumberOfPapers = np.sum([len(paperUIDs) for paperUIDs in self.year2IDs.values()])
        progressBar = tqdm(total=totalNumberOfPapers, disable=not self.showProgress)

        for year in analysisRange:
            edge2Weight = {}
            if(year-1 in self.year2IDs):
                previousYearNetwork = scYearlyNetworks[year-1]
                # from the network get the weights of the edges
                for edge in previousYearNetwork.es:
                    edgeIndices = (edge.source,edge.target)
                    edgeIndices = (min(edgeIndices),max(edgeIndices))
                    edge2Weight[edgeIndices] = edge["weight"]
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    progressBar.update(1)
                    subjects = self.data.loc[paperUID, 'subjects']
                    subjects = [subject2Index[subjectCategory] for subjectCategory in subjects]
                    subjects = list(set(subjects))
                    edges = list(combinations(subjects,2))
                    # make edges min(),max()
                    edges = set([(min(edge),max(edge)) for edge in edges])
                    edgesAndWeights.update(edges)
                    edgeWeights = [edge2Weight[edge]+1 for edge in edges if edge in edge2Weight]

                    VanguardNovelty = list(np.histogram(edgeWeights,bins=range(2,weightsCount+3))[0])
                    VanguardNoveltyByPaperID[paperUID] = tuple(VanguardNovelty)
                # year level
                edgeWeights = [edge2Weight[edge]+1 for edge in edgesAndWeights.keys() if edge in edge2Weight]
                VanguardNovelty = list(np.histogram(edgeWeights,bins=range(2,weightsCount+3))[0])
                VanguardNoveltyByYear[year] = tuple(VanguardNovelty)
            else:
                VanguardNoveltyByYear[year] = tuple([0]*weightsCount)
                # MaverickNoveltyByYear

        # sort according to VanguardNoveltyByPaperID via tuple sorting

        VanguardNoveltyByPaperIDSorted = [(k,v) for k, v in sorted(VanguardNoveltyByPaperID.items(), key=lambda item: item[1],reverse=True)]
        # create a rank for the VanguardNoveltyByPaperID. If there is a tie, the two papers should have the same rank
        VanguardNoveltyByPaperIDRank = {}
        currentRank = 0
        currentRankCount = 0
        previousVanguardNovelty = None
        for entryName,VanguardNovelty in VanguardNoveltyByPaperIDSorted:
            currentRankCount+=1
            if(previousVanguardNovelty!=VanguardNovelty):
                currentRank=currentRankCount
            VanguardNoveltyByPaperIDRank[entryName] = currentRank
            previousVanguardNovelty = VanguardNovelty

        # VanguardNoveltyByPaperIDRank = {entry[0]:index+1 for index,entry in enumerate(VanguardNoveltyByPaperIDSorted)}
        # paperUID2ReferencesSubjectCategories["WOS:A1996VL28900004"]

        # VanguardNoveltyByYearIDSorted = [(k,v) for k, v in sorted(VanguardNoveltyByYear.items(), key=lambda item: item[1],reverse=True)]
        # # create a rank for the VanguardNoveltyByYear. If there is a tie, the two papers should have the same rank
        # VanguardNoveltyByYearRank = {}
        # currentRank = 0
        # currentRankCount = 0
        # previousVanguardNovelty = None
        # for entryName,VanguardNovelty in VanguardNoveltyByYearIDSorted:
        #     currentRankCount+=1
        #     if(previousVanguardNovelty!=VanguardNovelty):
        #         currentRank=currentRankCount
        #     VanguardNoveltyByYearRank[entryName] = currentRank
        #     previousVanguardNovelty = VanguardNovelty
        # strengtheneNoveltiyRankByYear = {entry[0]:index+1 for index,entry in enumerate(VanguardNoveltyByYearIDSorted)}
    




        # %%
        window = 5
        progressBar = tqdm(total=totalNumberOfPapers, disable=not self.showProgress)
        VanguardNoveltyImpactByPaperID = {}
        VanguardNoveltyImpactByYear = {}
        for year in yearRange:
            edge2Weight = {}
            if(year-1 in self.year2IDs):
                previousYearNetwork = scYearlyNetworks[year-1]
                # from the network get the weights of the edges
                for edge in previousYearNetwork.es:
                    edgeIndices = (edge.source,edge.target)
                    edgeIndices = (min(edgeIndices),max(edgeIndices))
                    edge2Weight[edgeIndices] = edge["weight"]
            changedSubjectCategories = set()
            if(year in self.year2IDs):
                papers = self.year2IDs[year]
                for paperUID in papers:
                    progressBar.update(1)
                    subjects = self.data.loc[paperUID, 'subjects']
                    subjects = [subject2Index[subject] for subject in subjects]
                    subjects = list(set(subjects))
                    edges = list(combinations(subjects,2))
                    # make edges min(),max()
                    edges = set([(min(edge),max(edge)) for edge in edges])
                    edgesAndWeights.update(edges)
                    allowedEdges=[edge for edge in edges if edge in edge2Weight and edge2Weight[edge]+1<=weightsCount+1]
                    activatedSubjectCategories = set([subjectCategory for edge in allowedEdges for subjectCategory in edge])
                    changedSubjectCategories.update(activatedSubjectCategories)
                    
                    VanguardNoveltiyImpacts = []
                    for subjectCategory in activatedSubjectCategories:
                        yearBeforeCitations = cumulativeCitationsReceivedSCYear[year-1][subjectCategory] if year-1 in cumulativeCitationsReceivedSCYear else 0
                        windowYearsAfterCitations = cumulativeCitationsReceivedSCYear[year+window][subjectCategory] if year+window in cumulativeCitationsReceivedSCYear else 0
                        VanguardNoveltiyImpacts.append(windowYearsAfterCitations-yearBeforeCitations)
                    VanguardNoveltyImpactByPaperID[paperUID] = VanguardNoveltiyImpacts
                # year level
                VanguardNoveltiyImpacts = []
                for subjectCategory in changedSubjectCategories:
                    yearBeforeCitations = cumulativeCitationsReceivedSCYear[year-1][subjectCategory] if year-1 in cumulativeCitationsReceivedSCYear else 0
                    windowYearsAfterCitations = cumulativeCitationsReceivedSCYear[year+window][subjectCategory] if year+window in cumulativeCitationsReceivedSCYear else 0
                    VanguardNoveltiyImpacts.append(windowYearsAfterCitations-yearBeforeCitations)
                VanguardNoveltyImpactByYear[year] = VanguardNoveltiyImpacts
            else:
                VanguardNoveltyImpactByYear[year] = []
                # MaverickNoveltyByYear

        # VanguardNoveltyImpactByYearAverage = {year:np.average(VanguardNoveltyImpactByYear[year]) if VanguardNoveltyImpactByYear[year] else 0 for year in VanguardNoveltyImpactByYear}
        # VanguardNoveltyImpactByYearMedian = {year:np.median(VanguardNoveltyImpactByYear[year]) if VanguardNoveltyImpactByYear[year] else 0 for year in VanguardNoveltyImpactByYear}
        # VanguardNoveltyImpactByYearMax = {year:np.max(VanguardNoveltyImpactByYear[year]) if VanguardNoveltyImpactByYear[year] else 0 for year in VanguardNoveltyImpactByYear}
        # VanguardNoveltyImpactByYearMin = {year:np.min(VanguardNoveltyImpactByYear[year]) if VanguardNoveltyImpactByYear[year] else 0 for year in VanguardNoveltyImpactByYear}

        # scores by paper
        VanguardNoveltyImpactByPaperIDAverage = {paperUID:np.average(VanguardNoveltyImpactByPaperID[paperUID]) if VanguardNoveltyImpactByPaperID[paperUID] else 0 for paperUID in VanguardNoveltyImpactByPaperID}
        # VanguardNoveltyImpactByPaperIDMedian = {paperUID:np.median(VanguardNoveltyImpactByPaperID[paperUID]) if VanguardNoveltyImpactByPaperID[paperUID] else 0 for paperUID in VanguardNoveltyImpactByPaperID}
        # VanguardNoveltyImpactByPaperIDMax = {paperUID:np.max(VanguardNoveltyImpactByPaperID[paperUID]) if VanguardNoveltyImpactByPaperID[paperUID] else 0 for paperUID in VanguardNoveltyImpactByPaperID}
        # VanguardNoveltyImpactByPaperIDMin = {paperUID:np.min(VanguardNoveltyImpactByPaperID[paperUID]) if VanguardNoveltyImpactByPaperID[paperUID] else 0 for paperUID in VanguardNoveltyImpactByPaperID}



        publicationIDsList = list(self.data.index)
        tableData = {}
        tableData["publicationID"] = publicationIDsList
        tableData["VanguardNovelty"] = [VanguardNoveltyByPaperID[paperID] if paperID in VanguardNoveltyByPaperID else np.nan for paperID in publicationIDsList]
        tableData["VanguardNoveltyRank"] = [VanguardNoveltyByPaperIDRank[paperID] if paperID in VanguardNoveltyByPaperIDRank else np.nan for paperID in publicationIDsList]
        tableData["VanguardNoveltyImpact"] = [VanguardNoveltyImpactByPaperIDAverage[paperID] if paperID in VanguardNoveltyImpactByPaperIDAverage else np.nan for paperID in publicationIDsList]
        return pd.DataFrame(tableData)

    # creates a copy but with shuffled subjects
    def generateBaseModelInstance(self, attractiveness, showProgress=False):
        """
        Generate a base model instance with shuffled subject assignments.

        Parameters
        ----------
        attractiveness : float
            Attractiveness parameter for subject assignment probabilities.
        showProgress : bool, optional
            Whether to show progress bars during computation. Default is False.

        Returns
        -------
        CitationData
            A new CitationData instance with shuffled subjects.
        """
        # create a new model
        newModel = CitationData(self.data,
                                baselineRange=self.baselineRange,
                                analysisRange=self.analysisRange,
                                attractiveness=attractiveness,
                                showProgress=showProgress
                                )
                                
        return newModel



    def _shuffleSubjects(self):
        startingYear = self.baselineRange[1]
        coreSubjectCategories = set()
        allYears = sorted(list(self.year2IDs.keys()))
        coreYearRange = range(allYears[0],startingYear)
        fullYearRange = range(allYears[0],allYears[-1]+1)
        # first year after the baseline
        modelStartYear = startingYear


        subjectCategoryFrequencyCummulativePerYear = {}
        previousYearFrequency = Counter()
        # paperSubjectCategoryCountsForPaperPerYear = {}
        for year in fullYearRange:
            subjectCategories = Counter()
            subjectCategoriesCountsInPaper = []
            if year in self.year2IDs:
                paperUIDs = self.year2IDs[year]
                for paperUID in paperUIDs:
                    subjects = self.data.loc[paperUID, 'subjects']
                    subjectCategories.update(subjects)
                        # subjectCategoriesCountsInPaper.append(len(arrayOfSubjectCategories))

            previousYearFrequency.update(subjectCategories)
            subjectCategoryFrequencyCummulativePerYear[year] = previousYearFrequency.copy()
            # paperSubjectCategoryCountsPerYear[year] = subjectCategoriesCountsInPaper

        # %%


        # %%
        allSubjectCategories = list(set(previousYearFrequency.keys()))
        # convert subjectCategoryFrequencyCummulativePerYear to dictionary of arrays in the same order as allSubjectCategories
        subjectCategoryFrequencyCummulativePerYearArray = {}
        for year in fullYearRange:
            subjectCategoryFrequencyCummulativePerYearArray[year] = np.array([subjectCategoryFrequencyCummulativePerYear[year][subjectCategory] for subjectCategory in allSubjectCategories])


        # %%


        # %%
        # Model run
        totalNumberOfPapers = np.sum([len(paperUIDs) for paperUIDs in self.year2IDs.values()])
        modelPaperUID2ReferencesSubjectCategories = {}
        progressBar = tqdm(total=totalNumberOfPapers, disable=not self.showProgress)
        attractiveness = self.attractiveness
        for year in fullYearRange:
            if year in self.year2IDs:
                paperUIDs = self.year2IDs[year]
                if(year >= modelStartYear):
                    # Using subjectCategories
                    p = subjectCategoryFrequencyCummulativePerYearArray[year-1] + attractiveness
                    p = p/p.sum()
                for paperUID in paperUIDs:
                    progressBar.update(1)
                    subjects = self.data.loc[paperUID, 'subjects']
                    if(subjects):
                        if(year < modelStartYear):
                            modelArrayOfSubjectCategories = set(subjects)
                        else:
                            # run Model
                            modelArrayOfSubjectCategories = set()
                            subjectCategoryCount = len(subjects)
                            # select subjectCategoryCount from allSubjectCategories based on frequency until last year + attractiveness
                            # normalize
                            for i in range(subjectCategoryCount):
                                selectedSubjectCategoryIndex = np.random.choice(len(allSubjectCategories),1,p=p,replace=False)[0]
                                selectedSubjectCategory = allSubjectCategories[selectedSubjectCategoryIndex]
                                modelArrayOfSubjectCategories.add(selectedSubjectCategory)
                        modelPaperUID2ReferencesSubjectCategories[paperUID] = modelArrayOfSubjectCategories
                    # now we need to update the subjectCategoryFrequencyCummulativePerYearArray

        # %%
        modelSubjectCategoryReference2PaperUID = {}
        for philanthropyUID,subjectCategories in tqdm(modelPaperUID2ReferencesSubjectCategories.items(), disable=not self.showProgress):
            for subjectCategory in subjectCategories:
                if subjectCategory not in modelSubjectCategoryReference2PaperUID:
                    modelSubjectCategoryReference2PaperUID[subjectCategory] = set()
                modelSubjectCategoryReference2PaperUID[subjectCategory].add(philanthropyUID)

        modelSubjects = [list(modelPaperUID2ReferencesSubjectCategories[paperUID]) if paperUID in modelPaperUID2ReferencesSubjectCategories else [] for paperUID in self.data.index]
        self.data["subjects"] = modelSubjects
        self.data["referenceSubjects"] = modelSubjects