import random
from typing import List, Dict, Tuple

class GoEmotionsRubric:
    """
    Rubric for GoEmotions dataset evaluation with 28 emotion categories
    """
    
    def __init__(self):
        self.emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        # Emotion definitions for better understanding
        self.emotion_definitions = {
            'admiration': 'Feeling respect and approval for someone or something',
            'amusement': 'Finding something funny or entertaining',
            'anger': 'Strong feeling of annoyance, displeasure, or hostility',
            'annoyance': 'Feeling slightly angry or irritated',
            'approval': 'Having a favorable opinion; agreeing with something',
            'caring': 'Feeling concern or interest; showing kindness',
            'confusion': 'Unable to understand; being puzzled',
            'curiosity': 'Eager to know or learn something',
            'desire': 'Strong feeling of wanting something',
            'disappointment': 'Feeling sad because something did not meet expectations',
            'disapproval': 'Having an unfavorable opinion; disagreeing',
            'disgust': 'Strong feeling of dislike or revulsion',
            'embarrassment': 'Feeling ashamed or awkward',
            'excitement': 'Feeling enthusiastic and eager',
            'fear': 'Feeling afraid or anxious about something',
            'gratitude': 'Feeling thankful and appreciative',
            'grief': 'Deep sorrow, especially over loss',
            'joy': 'Feeling great pleasure and happiness',
            'love': 'Strong affection or deep caring for someone/something',
            'nervousness': 'Feeling anxious or uneasy',
            'optimism': 'Feeling hopeful and positive about the future',
            'pride': 'Feeling satisfaction in achievements or qualities',
            'realization': 'Becoming aware of something; understanding',
            'relief': 'Feeling reassured or relaxed after anxiety',
            'remorse': 'Feeling regret or guilt about something',
            'sadness': 'Feeling unhappy or sorrowful',
            'surprise': 'Feeling amazed or astonished by something unexpected',
            'neutral': 'Not expressing strong emotion; balanced'
        }
        
        # Contrastive examples for each emotion
        self.contrastive_examples = {
            'admiration': {
                'positive': ["I really admire how she handled that difficult situation", "His dedication to helping others is truly admirable"],
                'negative': ["I can't stand how arrogant he is", "That was a terrible way to treat people"]
            },
            'amusement': {
                'positive': ["That joke was hilarious!", "I couldn't stop laughing at that video"],
                'negative': ["This is so boring", "I don't find that funny at all"]
            },
            'anger': {
                'positive': ["I'm furious about this injustice!", "This makes me absolutely livid"],
                'negative': ["I'm so happy about this news", "This brings me such peace"]
            },
            'annoyance': {
                'positive': ["This is getting on my nerves", "I'm getting irritated by this"],
                'negative': ["This is so pleasant", "I'm enjoying this so much"]
            },
            'approval': {
                'positive': ["I completely agree with this decision", "This is exactly what we needed"],
                'negative': ["I strongly disagree with this", "This is completely wrong"]
            },
            'caring': {
                'positive': ["I'm worried about your wellbeing", "Let me know if you need anything"],
                'negative': ["I don't care what happens to you", "That's not my problem"]
            },
            'confusion': {
                'positive': ["I'm not sure I understand this", "This is puzzling to me"],
                'negative': ["This is crystal clear", "I understand this perfectly"]
            },
            'curiosity': {
                'positive': ["I wonder what happens next", "I'm curious about this topic"],
                'negative': ["I have no interest in this", "I don't want to know more"]
            },
            'desire': {
                'positive': ["I really want this to happen", "I wish I could have that"],
                'negative': ["I don't want this at all", "I hope this never happens"]
            },
            'disappointment': {
                'positive': ["I'm disappointed this didn't work out", "This fell short of expectations"],
                'negative': ["This exceeded my expectations", "I'm thrilled with how this turned out"]
            },
            'disapproval': {
                'positive': ["I don't approve of this behavior", "This is unacceptable"],
                'negative': ["I fully support this", "This is perfectly acceptable"]
            },
            'disgust': {
                'positive': ["This is revolting", "I find this absolutely disgusting"],
                'negative': ["This is beautiful", "I love this so much"]
            },
            'embarrassment': {
                'positive': ["I'm so embarrassed about what happened", "This is mortifying"],
                'negative': ["I'm proud of what I did", "This makes me feel confident"]
            },
            'excitement': {
                'positive': ["I'm so excited about this!", "This is thrilling!"],
                'negative': ["This is so boring", "I'm completely uninterested"]
            },
            'fear': {
                'positive': ["I'm scared about what might happen", "This is terrifying"],
                'negative': ["I feel completely safe", "This is reassuring"]
            },
            'gratitude': {
                'positive': ["I'm so grateful for your help", "Thank you so much for this"],
                'negative': ["I don't appreciate this", "This is not helpful at all"]
            },
            'grief': {
                'positive': ["I'm heartbroken about this loss", "This is devastating"],
                'negative': ["I'm celebrating this news", "This brings me great joy"]
            },
            'joy': {
                'positive': ["I'm overjoyed about this!", "This makes me so happy"],
                'negative': ["This makes me miserable", "I'm devastated by this"]
            },
            'love': {
                'positive': ["I love this so much", "This means everything to me"],
                'negative': ["I hate this", "This disgusts me"]
            },
            'nervousness': {
                'positive': ["I'm nervous about the presentation", "This makes me anxious"],
                'negative': ["I'm completely confident", "This is easy for me"]
            },
            'optimism': {
                'positive': ["I'm hopeful things will improve", "This will work out great"],
                'negative': ["I don't think this will work", "I'm pessimistic about this"]
            },
            'pride': {
                'positive': ["I'm proud of this achievement", "This is a great accomplishment"],
                'negative': ["I'm ashamed of this", "This is embarrassing"]
            },
            'realization': {
                'positive': ["I just realized what this means", "Now I understand"],
                'negative': ["I'm still confused", "This doesn't make sense"]
            },
            'relief': {
                'positive': ["I'm relieved that's over", "What a relief!"],
                'negative': ["I'm still worried", "This is still stressful"]
            },
            'remorse': {
                'positive': ["I regret doing that", "I feel guilty about this"],
                'negative': ["I'm glad I did that", "I have no regrets"]
            },
            'sadness': {
                'positive': ["This makes me sad", "I'm feeling down about this"],
                'negative': ["This makes me happy", "I'm feeling great about this"]
            },
            'surprise': {
                'positive': ["I'm surprised by this!", "This is unexpected"],
                'negative': ["This is exactly what I expected", "I saw this coming"]
            },
            'neutral': {
                'positive': ["This is fine", "It's okay"],
                'negative': ["This is terrible", "This is amazing"]
            }
        }
    
    def get_emotion_definition(self, emotion: str) -> str:
        """Get the definition of a specific emotion"""
        return self.emotion_definitions.get(emotion, "Unknown emotion")
    
    def get_all_emotions(self) -> List[str]:
        """Get list of all emotions"""
        return self.emotions.copy()
    
    def format_emotions_for_prompt(self) -> str:
        """Format emotions list for use in prompts"""
        return ", ".join(self.emotions)
    
    def get_emotion_examples(self, emotion: str, num_examples: int = 2) -> List[str]:
        """Get example texts for a specific emotion"""
        if emotion not in self.contrastive_examples:
            return []
        
        examples = self.contrastive_examples[emotion]['positive']
        return random.sample(examples, min(num_examples, len(examples)))
    
    def get_contrastive_examples(self, target_emotion: str, num_examples: int = 2) -> Dict[str, List[str]]:
        """
        Get contrastive examples for the target emotion
        Returns both positive examples (showing the emotion) and negative examples (not showing the emotion)
        """
        if target_emotion not in self.contrastive_examples:
            return {'positive': [], 'negative': []}
        
        examples = self.contrastive_examples[target_emotion]
        
        # Sample from available examples
        positive_examples = random.sample(
            examples['positive'], 
            min(num_examples, len(examples['positive']))
        )
        negative_examples = random.sample(
            examples['negative'], 
            min(num_examples, len(examples['negative']))
        )
        
        return {
            'positive': positive_examples,
            'negative': negative_examples
        }
    
    def get_similar_emotions(self, emotion: str) -> List[str]:
        """Get emotions that are similar to the given emotion"""
        similar_groups = {
            'joy': ['amusement', 'excitement', 'optimism'],
            'sadness': ['grief', 'disappointment', 'remorse'],
            'anger': ['annoyance', 'disgust', 'disapproval'],
            'fear': ['nervousness', 'confusion'],
            'love': ['admiration', 'caring', 'gratitude'],
            'surprise': ['realization', 'curiosity'],
            'pride': ['approval', 'admiration'],
            'relief': ['gratitude', 'joy'],
            'embarrassment': ['remorse', 'nervousness']
        }
        
        # Return similar emotions, or empty list if none found
        return similar_groups.get(emotion, [])
    
    def validate_emotion(self, emotion: str) -> bool:
        """Check if an emotion is valid"""
        return emotion in self.emotions
    
    def get_emotion_category(self, emotion: str) -> str:
        """Get the broad category of an emotion"""
        categories = {
            'positive': ['admiration', 'amusement', 'approval', 'caring', 'excitement', 
                        'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'],
            'negative': ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                        'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'],
            'neutral': ['confusion', 'curiosity', 'desire', 'realization', 'surprise', 'neutral']
        }
        
        for category, emotions in categories.items():
            if emotion in emotions:
                return category
        return 'unknown'
    
    def get_prompt_descriptions(self) -> Dict[str, str]:
        """
        Get prompt descriptions for all emotions - this is what your evaluation files need!
        Returns a dictionary mapping emotion names to their prompt descriptions
        """
        prompt_descriptions = {}
        
        for emotion in self.emotions:
            prompt_descriptions[emotion] = f"The comment expresses {emotion}: {self.emotion_definitions[emotion]}"
        
        return prompt_descriptions
    
    def parse_emotion_response(self, response_text: str) -> List[str]:
        """
        Parse emotion response from LLM output
        Tries to extract a list of emotions from various response formats
        """
        import re
        import ast
        
        # Clean the response
        response_text = response_text.strip()
        
        # Try to parse as Python list first
        try:
            # Look for list pattern like ['emotion1', 'emotion2']
            list_match = re.search(r'\[([^\]]+)\]', response_text)
            if list_match:
                list_str = '[' + list_match.group(1) + ']'
                parsed = ast.literal_eval(list_str)
                if isinstance(parsed, list):
                    return [str(item).strip().strip("'\"") for item in parsed]
        except:
            pass
        
        # Try to find emotions mentioned in the text
        found_emotions = []
        response_lower = response_text.lower()
        
        for emotion in self.emotions:
            if emotion.lower() in response_lower:
                found_emotions.append(emotion)
        
        return found_emotions
    
    def validate_emotions(self, emotions: List[str]) -> List[str]:
        """
        Validate that emotions are in the valid emotion set
        """
        valid_emotions = []
        for emotion in emotions:
            if emotion in self.emotions:
                valid_emotions.append(emotion)
        return valid_emotions