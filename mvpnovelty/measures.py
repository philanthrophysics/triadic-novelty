
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import igraph as ig
from collections import Counter
from itertools import combinations 
import sys
epsilon = sys.float_info.epsilon

class CitationData:
    # baselineRange is a tuple
    def __init__(self, data: pd.DataFrame, useReferencesSubjects: bool = True, baselineRange: tuple = (-1, -1)):
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
        
        # convert references to list of strings (split by ;) skip empty entries
        self.data['references'] = self.data['references'].apply(lambda x: [ref.strip() for ref in x.split(';') if ref.strip()])
        #subjects to list of strings (split by ;) skip empty entries
        self.data['subjects'] = self.data['subjects'].apply(lambda x: [sub.strip() for sub in x.split(';') if sub.strip()])
        # year to int
        self.data['year'] = self.data['year'].astype(int)
        self.data["referenceSubjects"] = self._subjectsFromReferences()

        if(useReferencesSubjects):
            self.data['subjects'] = self.data['referenceSubjects']
        

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
        for year, publicationID in zip(tqdm(self.data['year'], desc="Creating year to publicationID mapping"), self.data.index):
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
        for year,referencesList in zip(tqdm(self.data['year'], desc="Calculating total references per year"),self.data['references']):
            if year not in year2TotalReferences:
                year2TotalReferences[year] = 0
            if referencesList==referencesList:
                year2TotalReferences[year] += len(referencesList)
        return year2TotalReferences
    

    def _getReferenceSubject2IDs(self):
        # create a dictionary with subject as key and publicationID as value
        subject2IDs = {}
        for paperID, subjects in zip(tqdm(self.data.index, desc="Processing reference subjects"), self.data['referenceSubjects']):
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

    def calculatePioneerNoveltyScores(self, analysisRange: tuple = (-1, -1), impactWindowSize: int = 5):
        # calculate the pioneer novelty score
        # for each subject in the baseline range, get the number of references that were introduced in the baseline range
        # and divide by the total number of references
        # return a dictionary with pioneer novelty score for each paperID and the introduced subject categories
        paperID2PioneerNoveltyScore = {}
        paperID2IntroducedSubjects = {}

        analysisRange = self._checkAnalysisRange(analysisRange)
        
        coreSubjects = set()
        # prefill the coreSubjects with subjects introduced before the analysis range
        for year in range(self.baselineRange[0], analysisRange[0]):
            if year in self.year2IntroducedSubjects:
                coreSubjects.update(self.year2IntroducedSubjects[year])

        beforeYearSubjects = set(coreSubjects)
        for year in tqdm(range(analysisRange[0], analysisRange[1])):
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
        
        return {
            'subjectPioneerNoveltyTable': dfSubjectData,
            'paperPioneerNoveltyTable': dfPioneerNoveltyScores
        }


    def calculateShortenerNoveltyScores(self, analysisRange: tuple = (-1, -1), backwardWindow: int = 5, forwardWindow: int = 5):
        
        analysisRange = self._checkAnalysisRange(analysisRange)
        subjectsSet = set([subject for subjects in self.data.loc["subjects"] for subject in subjects])
        if "nan" in subjectsSet:
            subjectsSet.remove("nan")

        index2Subject = {index:subject for index,subject in enumerate(subjectsSet)}
        subject2Index = {subject:index for index,subject in enumerate(subjectsSet)}

        previousNetwork = ig.Graph(len(subject2Index),directed=False)


        scYearlyNetworks = {}
        paper2ShortenerNoveltyAverage = {}
        paper2ShortenerNoveltyMedian = {}
        paper2ShortenerNoveltyMax = {}

        year2ShortenerNoveltyAverage = {}
        year2ShortenerNoveltyMedian = {}
        year2ShortenerNoveltyMax = {}

        paper2ShortenerSubjects = {}

        # add edges to the graph based on the SC pairs in the references
        yearRange = range(analysisRange[0], analysisRange[1])
        for year in tqdm(yearRange):
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
                        subjectIndices = [subject2Index[subject] for subject in subjectIndices]
                        subjectIndices = list(set(subjectIndices))
                        edges = list(combinations(subjectIndices,2))
                        # make edges min(),max()
                        edges = [(min(edge),max(edge)) for edge in edges]
                        edgesSet = set(edges)
                        edgesAndWeights.update(edgesSet)
                        # previous distances between all new edges
                        noveltyComponents = []
                        for edge in edgesSet:
                            if(edge in distancesBuffer):
                                distance = distancesBuffer[edge]
                            else:
                                distance = previousNetwork.distances(source=edge[0],target=edge[1])[0][0]
                                distancesBuffer[edge] = distance
                            if(1.0-1.0/distance>epsilon): # check FIX #if(1-1.0/distance>0):
                                noveltyComponents.append(1-1.0/distance)
                                allYearEdge2NoveltyComponent[edge] = 1-1.0/distance
                                if(paperUID not in paper2ShortenerSubjects):
                                    paper2ShortenerSubjects[paperUID] = []
                                paper2ShortenerSubjects[paperUID].append((edge[0],edge[1],1-1.0/distance))

                        if(noveltyComponents):
                            paper2ShortenerNoveltyAverage[paperUID] = np.average(noveltyComponents)
                            paper2ShortenerNoveltyMedian[paperUID] = np.median(noveltyComponents)
                            paper2ShortenerNoveltyMax[paperUID] = np.max(noveltyComponents)
                        else:
                            paper2ShortenerNoveltyAverage[paperUID] = 0
                            paper2ShortenerNoveltyMedian[paperUID] = 0
                            paper2ShortenerNoveltyMax[paperUID] = 0
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
                year2ShortenerNoveltyAverage[year] = np.average(noveltyYearlyComponents)
                year2ShortenerNoveltyMedian[year] = np.median(noveltyYearlyComponents)
                year2ShortenerNoveltyMax[year] = np.max(noveltyYearlyComponents)
            else:
                year2ShortenerNoveltyAverage[year] = 0
                year2ShortenerNoveltyMedian[year] = 0
                year2ShortenerNoveltyMax[year] = 0
            
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
        for year in tqdm(yearRange):
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
        for year in tqdm(yearRange):
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

        shortenerNoveltyEnhancedAverageByYear = {}
        shortenerNoveltyDiminishAverageByYear = {}
        shortenerNoveltyImpactAverageByYear = {}

        shortenerNoveltyEnhancedAverageByPaperID = {}
        shortenerNoveltyDiminishAverageByPaperID = {}
        shortenerNoveltyImpactAverageByPaperID = {}

        shortenerNoveltyEnhancedMinByYear = {}
        shortenerNoveltyDiminishMinByYear = {}
        shortenerNoveltyImpactMinByYear = {}

        shortenerNoveltyEnhancedMinByPaperID = {}
        shortenerNoveltyDiminishMinByPaperID = {}
        shortenerNoveltyImpactMinByPaperID = {}

        shortenerNoveltyEnhancedMaxByYear = {}
        shortenerNoveltyDiminishMaxByYear = {}
        shortenerNoveltyImpactMaxByYear = {}

        shortenerNoveltyEnhancedMaxByPaperID = {}
        shortenerNoveltyDiminishMaxByPaperID = {}
        shortenerNoveltyImpactMaxByPaperID = {}

        shortenerNoveltyEnhancedMedianByYear = {}
        shortenerNoveltyDiminishMedianByYear = {}
        shortenerNoveltyImpactMedianByYear = {}

        shortenerNoveltyEnhancedMedianByPaperID = {}
        shortenerNoveltyDiminishMedianByPaperID = {}
        shortenerNoveltyImpactMedianByPaperID = {}

        with tqdm(total=len(self.data), desc="Calculating Shortener Novelty Scores") as progressBar:
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
                                shortenerNoveltyImpactAverageByPaperID[paperUID] = np.average(differences)
                                shortenerNoveltyImpactMinByPaperID[paperUID] = np.min(differences)
                                shortenerNoveltyImpactMaxByPaperID[paperUID] = np.max(differences)
                                shortenerNoveltyImpactMedianByPaperID[paperUID] = np.median(differences)
                            else:
                                shortenerNoveltyImpactAverageByPaperID[paperUID] = 0
                                shortenerNoveltyImpactMinByPaperID[paperUID] = 0
                                shortenerNoveltyImpactMaxByPaperID[paperUID] = 0
                                shortenerNoveltyImpactMedianByPaperID[paperUID] = 0
                            
                            # enhanced novelty for positives
                            positiveDifferences = [value for value in differences if value>0] # Check if this is correct
                            if(positiveDifferences):
                                shortenerNoveltyEnhancedAverageByPaperID[paperUID] = np.average(positiveDifferences)
                                shortenerNoveltyEnhancedMinByPaperID[paperUID] = np.min(positiveDifferences)
                                shortenerNoveltyEnhancedMaxByPaperID[paperUID] = np.max(positiveDifferences)
                                shortenerNoveltyEnhancedMedianByPaperID[paperUID] = np.median(positiveDifferences)
                            else:
                                shortenerNoveltyEnhancedAverageByPaperID[paperUID] = 0
                                shortenerNoveltyEnhancedMinByPaperID[paperUID] = 0
                                shortenerNoveltyEnhancedMaxByPaperID[paperUID] = 0
                                shortenerNoveltyEnhancedMedianByPaperID[paperUID] = 0
                            
                            # diminished novelty for negatives
                            negativeDifferences = [value for value in differences if value<0]
                            if(negativeDifferences):
                                shortenerNoveltyDiminishAverageByPaperID[paperUID] = np.average(negativeDifferences)
                                shortenerNoveltyDiminishMinByPaperID[paperUID] = np.min(negativeDifferences)
                                shortenerNoveltyDiminishMaxByPaperID[paperUID] = np.max(negativeDifferences)
                                shortenerNoveltyDiminishMedianByPaperID[paperUID] = np.median(negativeDifferences)
                            else:
                                shortenerNoveltyDiminishAverageByPaperID[paperUID] = 0
                                shortenerNoveltyDiminishMinByPaperID[paperUID] = 0
                                shortenerNoveltyDiminishMaxByPaperID[paperUID] = 0
                                shortenerNoveltyDiminishMedianByPaperID[paperUID] = 0

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
                        shortenerNoveltyImpactAverageByYear[year] = np.average(differences)
                        shortenerNoveltyImpactMinByYear[year] = np.min(differences)
                        shortenerNoveltyImpactMaxByYear[year] = np.max(differences)
                        shortenerNoveltyImpactMedianByYear[year] = np.median(differences)
                    else:
                        shortenerNoveltyImpactAverageByYear[year] = 0
                        shortenerNoveltyImpactMinByYear[year] = 0
                        shortenerNoveltyImpactMaxByYear[year] = 0
                        shortenerNoveltyImpactMedianByYear[year] = 0

                    # enhanced novelty for positives
                    positiveDifferences = [value for value in differences if value>=0]
                    if(positiveDifferences):
                        shortenerNoveltyEnhancedAverageByYear[year] = np.average(positiveDifferences)
                        shortenerNoveltyEnhancedMinByYear[year] = np.min(positiveDifferences)
                        shortenerNoveltyEnhancedMaxByYear[year] = np.max(positiveDifferences)
                        shortenerNoveltyEnhancedMedianByYear[year] = np.median(positiveDifferences)
                    else:
                        shortenerNoveltyEnhancedAverageByYear[year] = 0
                        shortenerNoveltyEnhancedMinByYear[year] = 0
                        shortenerNoveltyEnhancedMaxByYear[year] = 0
                        shortenerNoveltyEnhancedMedianByYear[year] = 0
                    
                    # diminished novelty for negatives
                    negativeDifferences = [value for value in differences if value<0]
                    if(negativeDifferences):
                        shortenerNoveltyDiminishAverageByYear[year] = np.average(negativeDifferences)
                        shortenerNoveltyDiminishMinByYear[year] = np.min(negativeDifferences)
                        shortenerNoveltyDiminishMaxByYear[year] = np.max(negativeDifferences)
                        shortenerNoveltyDiminishMedianByYear[year] = np.median(negativeDifferences)
                    else:
                        shortenerNoveltyDiminishAverageByYear[year] = 0
                        shortenerNoveltyDiminishMinByYear[year] = 0
                        shortenerNoveltyDiminishMaxByYear[year] = 0
                        shortenerNoveltyDiminishMedianByYear[year] = 0
                else:
                    shortenerNoveltyEnhancedAverageByYear[year] = 0
                    shortenerNoveltyDiminishAverageByYear[year] = 0
                    shortenerNoveltyEnhancedMinByYear[year] = 0
                    shortenerNoveltyDiminishMinByYear[year] = 0
                    shortenerNoveltyEnhancedMaxByYear[year] = 0
                    shortenerNoveltyDiminishMaxByYear[year] = 0
                    shortenerNoveltyEnhancedMedianByYear[year] = 0
                    shortenerNoveltyDiminishMedianByYear[year] = 0
                    shortenerNoveltyImpactAverageByYear[year] = 0
                    shortenerNoveltyImpactMinByYear[year] = 0
                    shortenerNoveltyImpactMaxByYear[year] = 0
                    shortenerNoveltyImpactMedianByYear[year] = 0
                    # shortenerNoveltyByYear
                accumulatedEdgesYearBefore = accumulatedEdgesYearBefore.union(edgesAndWeights.keys())
        # create tables for paperID
        publicationIDsList = list(self.data.index)
        #         paper2ShortenerNoveltyAverage = {}
        # paper2ShortenerNoveltyMedian = {}
        # paper2ShortenerNoveltyMax = {}

        # year2ShortenerNoveltyAverage = {}
        # year2ShortenerNoveltyMedian = {}
        # year2ShortenerNoveltyMax = {}
        tableData = {}
        tableData["publicationID"] = publicationIDsList
        tableData["shortenerNoveltyAverage"] = [paper2ShortenerNoveltyAverage[paperID] if paperID in paper2ShortenerNoveltyAverage else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyMedianList"] = [paper2ShortenerNoveltyMedian[paperID] if paperID in paper2ShortenerNoveltyMedian else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyMaxList"] = [paper2ShortenerNoveltyMax[paperID] if paperID in paper2ShortenerNoveltyMax else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltySubjectsList"] = [list(paper2ShortenerSubjects[paperID]) if paperID in paper2ShortenerSubjects else [] for paperID in publicationIDsList]

        # impacts
        tableData["shortenerNoveltyEnhancedAverage"] = [shortenerNoveltyEnhancedAverageByPaperID[paperID] if paperID in shortenerNoveltyEnhancedAverageByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyDiminishAverage"] = [shortenerNoveltyDiminishAverageByPaperID[paperID] if paperID in shortenerNoveltyDiminishAverageByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyImpactAverage"] = [shortenerNoveltyImpactAverageByPaperID[paperID] if paperID in shortenerNoveltyImpactAverageByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyEnhancedMin"] = [shortenerNoveltyEnhancedMinByPaperID[paperID] if paperID in shortenerNoveltyEnhancedMinByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyDiminishMin"] = [shortenerNoveltyDiminishMinByPaperID[paperID] if paperID in shortenerNoveltyDiminishMinByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyImpactMin"] = [shortenerNoveltyImpactMinByPaperID[paperID] if paperID in shortenerNoveltyImpactMinByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyEnhancedMax"] = [shortenerNoveltyEnhancedMaxByPaperID[paperID] if paperID in shortenerNoveltyEnhancedMaxByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyDiminishMax"] = [shortenerNoveltyDiminishMaxByPaperID[paperID] if paperID in shortenerNoveltyDiminishMaxByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyImpactMax"] = [shortenerNoveltyImpactMaxByPaperID[paperID] if paperID in shortenerNoveltyImpactMaxByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyEnhancedMedian"] = [shortenerNoveltyEnhancedMedianByPaperID[paperID] if paperID in shortenerNoveltyEnhancedMedianByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyDiminishMedian"] = [shortenerNoveltyDiminishMedianByPaperID[paperID] if paperID in shortenerNoveltyDiminishMedianByPaperID else np.nan for paperID in publicationIDsList]
        tableData["shortenerNoveltyImpactMedian"] = [shortenerNoveltyImpactMedianByPaperID[paperID] if paperID in shortenerNoveltyImpactMedianByPaperID else np.nan for paperID in publicationIDsList]
        dfShortenerNoveltyScores = pd.DataFrame(tableData)

        return dfShortenerNoveltyScores
