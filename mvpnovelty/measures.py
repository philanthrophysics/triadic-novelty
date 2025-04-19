
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

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
        # index by publicationID
        # make sure publicationID is a string
        self.data['publicationID'] = self.data['publicationID'].astype(str)
        self.data.set_index('publicationID', inplace=True)
        
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
            self.baselineRange = (baselineRange[0], self.data['year'].max())
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
        for year in self.data['year']:
            if year not in year2IDs:
                year2IDs[year] = []
            year2IDs[year].append(self.data.loc[year, 'publicationID'])
        return year2IDs
    
    def _generateSubject2IntroductionYear(self):
        # generate a dictionary with subject as key and introduction year as value
        subject2IntroductionYear = {}
        for year in self.year2IDs:
            for publicationID in self.year2IDs[year]:
                for subject in self.data.loc[publicationID, 'subjects']:
                    if subject not in subject2IntroductionYear:
                        subject2IntroductionYear[subject] = year
        return subject2IntroductionYear
    
    def _generateYear2IntroducedSubjects(self):
        # generate a dictionary with year as key and introduced subjects as value
        year2IntroducedSubjects = {}
        for year, subjects in self.subject2IntroductionYear.items():
            if year not in year2IntroducedSubjects:
                year2IntroducedSubjects[year] = []
            year2IntroducedSubjects[year].append(subjects)
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
        for year,referencesList in zip(tqdm(self.data['year']), self.data['references'], desc="Calculating total references per year"):
            if year not in year2TotalReferences:
                year2TotalReferences[year] = 0
            if referencesList==referencesList:
                year2TotalReferences[year] += len(referencesList.split("; "))
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


    def calculatePioneerNoveltyScores(self, analysisRange: tuple = (-1, -1), impactWindowSize: int = 5):
        # calculate the pioneer novelty score
        # for each subject in the baseline range, get the number of references that were introduced in the baseline range
        # and divide by the total number of references
        # return a dictionary with pioneer novelty score for each paperID and the introduced subject categories
        paperID2PioneerNoveltyScore = {}
        paperID2IntroducedSubjectCategories = {}
        if analysisRange[0] == -1:
            analysisRange = (self.baselineRange[1], analysisRange[1])
        if analysisRange[1] == -1:
            analysisRange = (analysisRange[0], self.data['year'].max())
        if analysisRange[0] > analysisRange[1]:
            raise ValueError("Analysis range is invalid")
        
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
                paperID2IntroducedSubjectCategories[paperID] = introducedSubjects
                paperID2PioneerNoveltyScore[paperID] = len(introducedSubjects)
            beforeYearSubjects.update(addedSubjectsOnYear)


        # Pioneer Novelty Impact
        # Subject category Novelty Impact
        subject2NoveltyImpact = {} # PapersReferencing(timewindow,SC1)/TotalPublications(timeWindow)
        for introducingYear,introducedSubjects in self.year2IntroducedSubjects:
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
            # dfSubjectCategoryData["introductionYear"] = [self.subject2IntroductionYear[subject] if subject in subjectCategory2IntroductionYear else np.NaN for subject in dfSubjectCategoryData["subject"]]
            dfSubjectData["introductionYear"] = [self.subject2IntroductionYear[subject] if subject in self.subject2IntroductionYear else np.NaN for subject in dfSubjectData["subject"]]
            # Set property NoveltyImpact
            # dfSubjectCategoryData[f"NoveltyImpact_W{impactWindowSize}"] = [subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow][subjectCategory] if subjectCategory in subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow] else np.NaN for subjectCategory in dfSubjectCategoryData["SubjectCategory"]]
            dfSubjectData["noveltyImpact_W" + str(impactWindowSize)] = [subject2NoveltyImpact[subject] if subject in subject2NoveltyImpact else np.NaN for subject in dfSubjectData["subject"]]
            # order by year
            dfSubjectData["introductionYear"] = dfSubjectData["introductionYear"].fillna(-1)
            dfSubjectData.sort_values(by=["introductionYear"],inplace=True,ascending=True)
            # set back nan values
            dfSubjectData["introductionYear"] = dfSubjectData["introductionYear"].replace(-1,np.NaN)
            # reset index
            dfSubjectData.reset_index(drop=True,inplace=True)
            

            paperUID2PioneerNoveltyImpactByWindow = {}
            paperUID2PioneerNoveltyImpactScoresByWindow = {}
            for pioneerNoveltyImpactTimeWindow in pioneerNoveltyImpactTimeWindows:
                paperUID2PioneerNoveltyImpact = {}
                paperUID2PioneerNoveltyImpactScores = {}
                for paperUID in dfPhilanthropy["UT"]:
                    if paperUID in paperUIDIntroducedSubjectCategories:
                        introducedSubjectCategories = paperUIDIntroducedSubjectCategories[paperUID]
                        impactScores = []
                        for subjectCategory in introducedSubjectCategories:
                            if subjectCategory in subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow]:
                                impactScores.append(subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow][subjectCategory])
                        if(len(impactScores)):
                            maxImpact = np.nanmax(impactScores)
                        else:
                            maxImpact = np.NaN
                        paperUID2PioneerNoveltyImpact[paperUID] = maxImpact
                        paperUID2PioneerNoveltyImpactScores[paperUID] = impactScores
                paperUID2PioneerNoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow] = paperUID2PioneerNoveltyImpact
                paperUID2PioneerNoveltyImpactScoresByWindow[pioneerNoveltyImpactTimeWindow] = paperUID2PioneerNoveltyImpactScores

            
            #             totalPaperCount+=len(year2paperUIDs[year])
            #     allPapersReferencing = subjectCategoryReference2PaperUID[subjectCategory]
            #     windowPapersReferencing = set()
            #     for paperUID in allPapersReferencing:
            #         if paperUID2Year[paperUID] in windowYearRange:
            #             windowPapersReferencing.add(paperUID)
            #     subjectCategory2NoveltyImpact[subjectCategory] = len(windowPapersReferencing)/totalPaperCount
            # subjectCategory2NoveltyImpactByWindow[pioneerNoveltyImpactTimeWindow] = subjectCategory2NoveltyImpact



        return {
            'pioneerScore': paperID2PioneerNoveltyScore,
            'introducedSubjects': paperID2IntroducedSubjectCategories,

        }



