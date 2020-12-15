import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.wrappers import LdaMallet
from scipy.spatial import distance
from processor.cleaner import prepare_text, seperate_genres

class AnimeRecommender:
    def __init__(self):
        """
        Anime recommender model.
        """

        try:
            self.doc_topic_dist = np.load('./model_files/doc_topic_dist.npy')
            self.spacy_df = pd.read_pickle('./model_files/spacy_df.pkl')
            self.lda_model = LdaMallet.load('./model_files/lda.model')
            self.dictionary = Dictionary.load('./model_files/dictionary')
        except:
            raise Exception("Unable to load all necessary files for the recommender system.")

    def __get_js_distances(self, title_index=None, distribution=None, k=10):
        """
        This function implements the Jensen-Shannon distance above
        and returns the top k indices of the smallest jensen shannon distances
        """

        query = None
        if title_index is not None:
            query = self.doc_topic_dist[title_index]
        elif distribution is not None:
            query = distribution

        sim=[distance.jensenshannon(data,query) for data in self.doc_topic_dist]
        return sim

    def __get_title_from_index(self, index):
        """
        Returns a title from an index in a dataframe
        
        Parameters
        ----------
        index: int
            The index of the title.
        """
        
        title = self.spacy_df[self.spacy_df.index == index]['Title'].values[0]
        return title

    def __get_index_from_title(self, title):
        """
        Returns a title's index in a dataframe
        
        Parameters
        ----------
        title: string
            Title to find its index.
        """
        
        index = self.spacy_df[self.spacy_df['Title'] == title].index.values[0]
        return index

    def recommend(self, title = None, synopsis = None, synopsis_genres = None, measure="similarity", k=10):
        """
        Returns recommendations from the topic similarity model.

        Parameters
        ----------
        title: string
            Title to base recommendations on
        synopsis: string
            Synopsis to base recommendations on
        synopsis_genres: string
            Comma seperated list of genres for the synopsis
        measure: string
            Either `similarity` or `distance` based on if you want recommendations of the
            most similar or different animes.
        k: int
            The number of recommendations to return
        """
        
        if title is None and synopsis is None:
            return
        
        score_indexes = []
        title_indexes = []
        
        if title is not None:
            title_index = self.__get_index_from_title(title)
            jensen_shannon_scores = self.__get_js_distances(title_index = title_index)
            most_similar_indexes = np.array(jensen_shannon_scores).argsort()
            
            if measure == 'similarity':
                sorted_js_scores = np.array(sorted(jensen_shannon_scores, reverse=True))
                score_indexes = sorted_js_scores[1:k + 1]
                title_indexes = most_similar_indexes[1:k + 1]

            elif measure == 'distance':
                sorted_js_scores = np.array(sorted(jensen_shannon_scores, reverse=False))
                score_indexes = sorted_js_scores[:k]
                title_indexes = most_similar_indexes[::-1][:k]
            
        elif synopsis is not None:
            cleaned_synopsis = prepare_text(synopsis)
            
            if synopsis_genres is not None:
                cleaned_synopsis = cleaned_synopsis + ' ' + seperate_genres(synopsis_genres)
                
            cleaned_synopsis_split = cleaned_synopsis.split()
            synopsis_bow = self.dictionary.doc2bow(cleaned_synopsis_split)
            synopsis_distribution = np.array([tup[1] for tup in self.lda_model[synopsis_bow]])
            jensen_shannon_scores = get_js_distances(distribution = synopsis_distribution)
            most_similar_indexes = np.array(jensen_shannon_scores).argsort()
            
            if measure == 'similarity':
                sorted_js_scores = np.array(sorted(jensen_shannon_scores, reverse=True))
                score_indexes = sorted_js_scores[1:k + 1]
                title_indexes = most_similar_indexes[1:k + 1]
            elif measure == 'distance':
                sorted_js_scores = np.array(sorted(jensen_shannon_scores, reverse=False))
                score_indexes = sorted_js_scores[:k]
                title_indexes = most_similar_indexes[::-1][:k]
        
        # (title name, jensen shannon score, topic distribution)
        recommendation_details = [(self.__get_title_from_index(title_idx), score_indexes[idx], self.doc_topic_dist[title_idx].tolist()) for idx, title_idx in enumerate(title_indexes)]
        return recommendation_details