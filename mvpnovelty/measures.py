
import pandas as pd
from tqdm.auto import tqdm


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
        if(useReferencesSubjects):
            self.data['subjects'] = self._subjectsFromReferences()

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
        for year,referencesList in zip(tqdm(self.data['year']), self.data['references']):
            if year not in year2TotalReferences:
                year2TotalReferences[year] = 0
            if referencesList==referencesList:
                year2TotalReferences[year] += len(referencesList.split("; "))
        return year2TotalReferences
    
    def calculatePioneerNoveltyScore(self, analysisRange: tuple = (-1, -1)):
        # calculate the pioneer novelty score
        # for each subject in the baseline range, get the number of references that were introduced in the baseline range
        # and divide by the total number of references
        # return a dictionary with subject as key and pioneer novelty score as value
        paperID2PioneerNoveltyScore = {}
        paperID2IntroducedSubjectCategories = {}
        if analysisRange[0] == -1:
            analysisRange = (self.baselineRange[1], analysisRange[1])
        if analysisRange[1] == -1:
            analysisRange = (analysisRange[0], self.data['year'].max())
        if analysisRange[0] > analysisRange[1]:
            raise ValueError("Analysis range is invalid")
        
        beforeYearSubjectCategories = set()
        for year in tqdm(range(analysisRange[0], analysisRange[1])):
            for paperID in self.year2IDs[year]:
                subjects = self.data.loc[paperID, 'subjects']
                introducedSubjects = set(subjects) - beforeYearSubjectCategories
                paperID2PioneerNoveltyScore