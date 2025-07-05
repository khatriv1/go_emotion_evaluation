# goemotions_evaluation/utils/emotion_rubric.py

"""
GoEmotions Rubric Implementation - 28 Categories for Reddit Comment Emotion Classification
Based on "GoEmotions: A Dataset of Fine-Grained Emotions"
"""

class GoEmotionsRubric:
    """
    GoEmotions' 28-category rubric for classifying emotions in Reddit comments.
    Categories derived from psychological literature and expert annotation.
    """
    
    @staticmethod
    def get_category_definitions():
        """Return definitions for all 28 GoEmotions categories."""
        return {
            "admiration": {
                "description": "Finding something impressive or worthy of respect",
                "examples": [
                    "That was an amazing performance!",
                    "I really respect what you accomplished",
                    "Wow, that's incredible work"
                ]
            },
            
            "amusement": {
                "description": "Finding something funny or entertaining",
                "examples": [
                    "Haha that made me laugh",
                    "This is hilarious!",
                    "LOL that's so funny"
                ]
            },
            
            "anger": {
                "description": "Strong feeling of displeasure or antagonism",
                "examples": [
                    "This makes me furious!",
                    "I'm so angry about this",
                    "That's absolutely infuriating"
                ]
            },
            
            "annoyance": {
                "description": "Mild anger, irritation",
                "examples": [
                    "This is getting on my nerves",
                    "How annoying",
                    "That's really bothering me"
                ]
            },
            
            "approval": {
                "description": "Having or expressing a favorable opinion",
                "examples": [
                    "I agree with this completely",
                    "That sounds good to me",
                    "Yes, exactly right"
                ]
            },
            
            "caring": {
                "description": "Displaying kindness and concern for others",
                "examples": [
                    "I hope you feel better soon",
                    "Take care of yourself",
                    "Sending you my support"
                ]
            },
            
            "confusion": {
                "description": "Lack of understanding, uncertainty",
                "examples": [
                    "I don't understand this at all",
                    "This is really confusing",
                    "What does this even mean?"
                ]
            },
            
            "curiosity": {
                "description": "Strong desire to know or learn something",
                "examples": [
                    "I wonder what happened next",
                    "Can you tell me more about this?",
                    "I'm really interested to know"
                ]
            },
            
            "desire": {
                "description": "Strong feeling of wanting something or wishing for something to happen",
                "examples": [
                    "I really want to try that",
                    "I wish I could be there",
                    "That looks so tempting"
                ]
            },
            
            "disappointment": {
                "description": "Sadness or displeasure caused by non-fulfillment of hopes or expectations",
                "examples": [
                    "I expected so much better",
                    "This is really disappointing",
                    "I'm let down by this result"
                ]
            },
            
            "disapproval": {
                "description": "Having or expressing an unfavorable opinion",
                "examples": [
                    "I don't agree with this at all",
                    "This is completely wrong",
                    "I disapprove of this decision"
                ]
            },
            
            "disgust": {
                "description": "Revulsion or strong disapproval aroused by something unpleasant",
                "examples": [
                    "That's absolutely disgusting",
                    "This makes me sick",
                    "How revolting"
                ]
            },
            
            "embarrassment": {
                "description": "Self-consciousness, shame, or awkwardness",
                "examples": [
                    "I feel so embarrassed about this",
                    "That's really awkward",
                    "I'm mortified"
                ]
            },
            
            "excitement": {
                "description": "Feeling of great enthusiasm and eagerness",
                "examples": [
                    "I can't wait for this!",
                    "This is so exciting!",
                    "I'm thrilled about this"
                ]
            },
            
            "fear": {
                "description": "Being afraid or worried",
                "examples": [
                    "I'm really scared about this",
                    "This worries me so much",
                    "I'm terrified of what might happen"
                ]
            },
            
            "gratitude": {
                "description": "Feeling of thankfulness and appreciation",
                "examples": [
                    "Thank you so much for this",
                    "I really appreciate your help",
                    "I'm so grateful for everything"
                ]
            },
            
            "grief": {
                "description": "Intense sorrow, especially caused by someone's death",
                "examples": [
                    "I miss them so much",
                    "This loss is devastating",
                    "My heart is broken"
                ]
            },
            
            "joy": {
                "description": "Feeling of pleasure and happiness",
                "examples": [
                    "I'm so happy about this!",
                    "This brings me such joy",
                    "I feel absolutely delighted"
                ]
            },
            
            "love": {
                "description": "Strong positive emotion of regard and affection",
                "examples": [
                    "I love this so much",
                    "You mean everything to me",
                    "This fills my heart with love"
                ]
            },
            
            "nervousness": {
                "description": "Apprehension, worry, anxiety",
                "examples": [
                    "I'm nervous about tomorrow",
                    "This makes me really anxious",
                    "I feel so uneasy about this"
                ]
            },
            
            "optimism": {
                "description": "Hopefulness and confidence about the future or success of something",
                "examples": [
                    "Things will definitely get better",
                    "I'm confident this will work out",
                    "I have a good feeling about this"
                ]
            },
            
            "pride": {
                "description": "Pleasure or satisfaction due to one's own achievements",
                "examples": [
                    "I'm so proud of what I accomplished",
                    "This achievement makes me proud",
                    "I feel great about my success"
                ]
            },
            
            "realization": {
                "description": "Becoming aware of something",
                "examples": [
                    "I just realized something important",
                    "Now I finally understand",
                    "It suddenly all makes sense"
                ]
            },
            
            "relief": {
                "description": "Reassurance and relaxation following release from anxiety or distress",
                "examples": [
                    "I'm so relieved it's over",
                    "Thank goodness that worked out",
                    "What a relief!"
                ]
            },
            
            "remorse": {
                "description": "Regret or guilty feeling",
                "examples": [
                    "I really regret doing that",
                    "I feel so bad about what happened",
                    "I'm sorry for my mistake"
                ]
            },
            
            "sadness": {
                "description": "Emotional pain, sorrow",
                "examples": [
                    "This makes me so sad",
                    "I feel really down about this",
                    "My heart feels heavy"
                ]
            },
            
            "surprise": {
                "description": "Feeling astonished, startled by something unexpected",
                "examples": [
                    "Wow, I didn't expect that!",
                    "What a surprise!",
                    "I'm shocked by this news"
                ]
            },
            
            "neutral": {
                "description": "No particular emotion expressed",
                "examples": [
                    "The meeting is scheduled for 3pm",
                    "Here is the requested information",
                    "The weather forecast shows rain"
                ]
            }
        }
    
    @staticmethod
    def get_prompt_descriptions():
        """Return prompt-friendly descriptions for each category."""
        return {
            "admiration": "The comment expresses admiration, respect, or finds something impressive or worthy of respect.",
            
            "amusement": "The comment finds something funny, entertaining, or expresses laughter/humor.",
            
            "anger": "The comment expresses strong displeasure, rage, fury, or antagonism.",
            
            "annoyance": "The comment expresses mild anger, irritation, or being bothered by something.",
            
            "approval": "The comment expresses agreement, endorsement, or a favorable opinion about something.",
            
            "caring": "The comment displays kindness, concern, compassion, or care for others.",
            
            "confusion": "The comment expresses lack of understanding, uncertainty, or being puzzled about something.",
            
            "curiosity": "The comment expresses interest, wonder, or a desire to know or learn more about something.",
            
            "desire": "The comment expresses wanting, wishing, craving, or longing for something to happen or to have something.",
            
            "disappointment": "The comment expresses sadness or displeasure caused by unmet expectations or hopes.",
            
            "disapproval": "The comment expresses disagreement, criticism, or an unfavorable opinion about something.",
            
            "disgust": "The comment expresses revulsion, strong disapproval, or finding something repulsive or offensive.",
            
            "embarrassment": "The comment expresses shame, awkwardness, self-consciousness, or feeling mortified.",
            
            "excitement": "The comment expresses enthusiasm, eagerness, thrill, or high energy about something.",
            
            "fear": "The comment expresses being afraid, scared, worried, or anxious about something.",
            
            "gratitude": "The comment expresses thankfulness, appreciation, or being grateful for something.",
            
            "grief": "The comment expresses intense sorrow, mourning, or sadness, especially related to loss or death.",
            
            "joy": "The comment expresses happiness, delight, pleasure, or positive feelings.",
            
            "love": "The comment expresses strong positive emotions, affection, or deep care for someone or something.",
            
            "nervousness": "The comment expresses anxiety, worry, uneasiness, or apprehension about something.",
            
            "optimism": "The comment expresses hopefulness, confidence, or positive expectations about the future.",
            
            "pride": "The comment expresses satisfaction, accomplishment, or positive feelings about achievements.",
            
            "realization": "The comment expresses suddenly understanding, becoming aware of, or recognizing something.",
            
            "relief": "The comment expresses feeling better after stress, worry, or difficulty has been resolved.",
            
            "remorse": "The comment expresses regret, guilt, or feeling sorry about something done or said.",
            
            "sadness": "The comment expresses sorrow, melancholy, feeling down, or emotional pain.",
            
            "surprise": "The comment expresses being astonished, shocked, or startled by something unexpected.",
            
            "neutral": "The comment does not express any particular emotion and is factual, informational, or emotionally neutral."
        }
    
    @staticmethod
    def get_emotion_groups():
        """Return emotion groups for analysis."""
        return {
            "positive": [
                'admiration', 'amusement', 'approval', 'caring', 'excitement', 
                'gratitude', 'joy', 'love', 'optimism', 'pride', 'relief'
            ],
            "negative": [
                'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
                'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness'
            ],
            "cognitive": [
                'confusion', 'curiosity', 'realization', 'surprise'
            ],
            "ambiguous": [
                'desire', 'neutral'
            ]
        }
    
    @staticmethod
    def get_all_emotions():
        """Return list of all 28 emotions."""
        return [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]

    def get_contrastive_examples(self, emotion: str) -> tuple[list[str], list[str]]:
            """
            Return a tuple (positive_examples, negative_examples) for a given emotion.
            Positive examples show that emotion; negatives show other emotions.
            """
            defs = self.get_category_definitions()
            if emotion not in defs:
                raise KeyError(f"Unknown emotion: {emotion}")

            # 1) Take up to N positive examples for this emotion
            pos = defs[emotion]["examples"][:3]

            # 2) Collect one example each from a few other emotions as negatives
            neg = []
            for emo, data in defs.items():
                if emo == emotion:
                    continue
                # grab one example from each until we have, say, 3 negatives
                if data["examples"]:
                    neg.append(data["examples"][0])
                if len(neg) >= 3:
                    break

            return pos, neg
# For easy import - same pattern as SIGHT
EMOTION_CATEGORIES = GoEmotionsRubric.get_all_emotions()
EMOTION_DEFINITIONS = GoEmotionsRubric.get_category_definitions()
EMOTION_PROMPT_DESCRIPTIONS = GoEmotionsRubric.get_prompt_descriptions()
EMOTION_GROUPS = GoEmotionsRubric.get_emotion_groups()