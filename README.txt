Instrctions for Running Scripts

1. Data-Aid Front End.html is front end web page of Jupyter notebook
(see ipynb file and run in jupyter notebook for more)
2. Downlaod latest version of R, RStudio and all packages in 
datasetPreProcess.r
AssociationRulesWithUCIDataset.r

3. Download Python 3.8 and all libraries used in the .py scripts 
Project used PIP to download and manage Python packages

4. Run AssociationRulesWithUCIDataset.r for Association Rules Data Mining (ensure working directory is set correctly)

5. Run all files with UT.py at the end to run the unit tests (see document for outline of unit tests)

6. Run KMeans.py for KMeans Model
7. Run sklearnModel.py for quick LDA model
7a. Run TopicEval.py for longer LDA model with hyperparam testing

8. For Minisom.py, youll need glove vectors, see GLOVEDOWNLOADLINK txt file for link and instructions

9. For Google Vision implemnentations youll need my credential files to be in folder, ensure that is there

10. to run on all receipt images, run loopImages.py, data will save to text.csv
11. run STTM.py to see LDA model on that data



NOTE: Paths may be different on your PC so you may need to change paths where apppropriate