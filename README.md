# GoEmotions Multi-Label Emotion Classification Evaluation

This project evaluates different prompting techniques for classifying emotions in Reddit comments using the GoEmotions dataset. It compares nine advanced LLM prompting strategies to determine which best matches human expert emotion annotations in multi-label emotion classification.

## Project Overview

The project classifies Reddit comments into 28 fine-grained emotion categories:
- **Positive emotions**: admiration, amusement, approval, caring, excitement, gratitude, joy, love, optimism, pride, relief
- **Negative emotions**: anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
- **Cognitive emotions**: confusion, curiosity, realization, surprise
- **Ambiguous emotions**: desire, neutral

### Multi-Label Classification Approach
The system uses multi-label classification where:
- **Human experts** provide binary labels (1/0) for each of the 28 emotions
- **AI models** predict emotion lists for each comment
- **Fair comparison** between human and AI emotion sets
- **Multiple emotions** per comment are supported (e.g., joy + excitement + optimism)

## Prompting Techniques Evaluated

1. **Zero-shot**: Direct classification using emotion definitions
2. **Chain of Thought (CoT)**: Step-by-step reasoning before classification
3. **Few-shot**: Provides emotion examples before asking for classification
4. **Active Prompting**: Selects most informative examples using uncertainty sampling
5. **Auto-CoT**: Automatically generates reasoning chains for emotions
6. **Contrastive CoT**: Uses positive and negative reasoning
7. **Rephrase and Respond**: Rephrases comment for clarity before classification
8. **Self-Consistency**: Multiple reasoning paths with majority voting
9. **Take a Step Back**: Derives emotion principles before classification

## Directory Structure

```
goemotions_evaluation/
├── data/
│   └── goemotions_1.csv       
├── prompting/
│   ├── zero_shot.py           
│   ├── cot.py                 
│   ├── few_shot.py            
│   ├── active_prompt.py       
│   ├── auto_cot.py            
│   ├── contrastive_cot.py     
│   ├── rephrase_and_respond.py 
│   ├── self_consistency.py    
│   └── take_a_step_back.py    
├── utils/
│   ├── data_loader.py       
│   ├── emotion_rubric.py     
│   └── metrics.py            
├── evaluation/
│   ├── evaluate_zero_shot.py
│   ├── evaluate_cot.py
│   ├── evaluate_few_shot.py
│   ├── evaluate_active_prompt.py
│   ├── evaluate_auto_cot.py
│   ├── evaluate_contrastive_cot.py
│   ├── evaluate_rephrase_respond.py
│   ├── evaluate_self_consistency.py
│   └── evaluate_take_step_back.py
├── results/                   
├── config.py                  
├── main.py                    
└── requirements.txt           
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/khatriv1/go_emotion_evaluation.git
cd go_emotion_evaluation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Configure your OpenAI API key in `config.py`:
```python
OPENAI_API_KEY = "your-api-key-here"
```

5. Ensure the GoEmotions dataset is in the data directory as `goemotions_1.csv`.

## Usage

### Run Complete Evaluation
```bash
python main.py
```

You'll be prompted to:
1. Enter number of comments to evaluate (recommended: 10-50 for testing)
2. Select which techniques to evaluate or run all

### Run Individual Technique
```bash
python evaluation/evaluate_zero_shot.py
```

## Dataset Format

The GoEmotions dataset contains Reddit comments with expert emotion annotations. Each comment can express multiple emotions simultaneously. The data loader:
- Processes binary emotion labels (0/1) for 28 emotions
- Converts labels to emotion lists for multi-label classification
- Handles multi-label evaluation metrics

## Multi-Label Classification

The GoEmotions dataset allows multiple emotions per comment:
- A comment can express joy + excitement + admiration simultaneously
- Evaluation requires exact matching of emotion sets
- Specialized metrics account for partial matches

## Evaluation Metrics

The project uses 6 key metrics for multi-label evaluation:

1. **Exact Match Accuracy**: Percentage where all emotions match exactly
2. **Cohen's Kappa (κ)**: Agreement beyond chance (-1 to 1, higher is better)
3. **Krippendorff's Alpha (α)**: Reliability measure (0 to 1)
4. **Intraclass Correlation (ICC)**: Pattern correlation between human and AI
5. **Hamming Loss**: Average per-label classification error (lower is better)
6. **Subset Accuracy**: Jaccard similarity of emotion sets (0 to 1, higher is better)

## Output

Results are saved in timestamped directories containing:
- `goemotions_all_techniques_comparison.csv` - Overall metrics comparison
- `goemotions_metrics_comparison.png` - Visual comparison chart
- `goemotions_all_detailed_results.csv` - All predictions
- `goemotions_comprehensive_report.txt` - Detailed analysis
- Individual technique results in subdirectories

## Key Features

- **Multi-label Classification**: Comments can express multiple emotions
- **28 Fine-grained Emotions**: Comprehensive emotion taxonomy
- **Fair Evaluation**: AI only sees comments, not human labels
- **Comprehensive Metrics**: Multiple ways to measure multi-label performance
- **Flexible Execution**: Run all or selected techniques

## Requirements

- Python 3.7+
- OpenAI API key with GPT-3.5 access
- Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, krippendorff, openai

## Notes

- Minimum 10 comments recommended for meaningful results
- Processing time depends on number of comments and techniques selected
- API rate limits are handled automatically
- Multi-label classification is more challenging than single-label

## Citation

If you use this code for your publication, please cite the original paper:

```
@inproceedings{demszky2020goemotions,
  author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},
  year = {2020}
}
```

## Dataset License

The GoEmotions dataset is licensed under the Apache License 2.0. Please ensure compliance with the license terms when using this dataset for research or educational purposes.

## Emotion Categories

### Positive Emotions (11)
- **admiration**: Finding something impressive or worthy of respect
- **amusement**: Finding something funny or entertaining  
- **approval**: Having or expressing a favorable opinion
- **caring**: Displaying kindness and concern for others
- **excitement**: Feeling of great enthusiasm and eagerness
- **gratitude**: Feeling of thankfulness and appreciation
- **joy**: Feeling of pleasure and happiness
- **love**: Strong positive emotion of regard and affection
- **optimism**: Hopefulness and confidence about the future
- **pride**: Pleasure or satisfaction due to achievements
- **relief**: Reassurance following release from anxiety

### Negative Emotions (11)
- **anger**: Strong feeling of displeasure or antagonism
- **annoyance**: Mild anger, irritation
- **disappointment**: Sadness caused by non-fulfillment of hopes
- **disapproval**: Having or expressing an unfavorable opinion
- **disgust**: Revulsion or strong disapproval
- **embarrassment**: Self-consciousness, shame, or awkwardness
- **fear**: Being afraid or worried
- **grief**: Intense sorrow, especially from loss
- **nervousness**: Apprehension, worry, anxiety
- **remorse**: Regret or guilty feeling
- **sadness**: Emotional pain, sorrow

### Cognitive Emotions (4)
- **confusion**: Lack of understanding, uncertainty
- **curiosity**: Strong desire to know or learn something
- **realization**: Becoming aware of something
- **surprise**: Being astonished by something unexpected

### Ambiguous (2)
- **desire**: Strong feeling of wanting something
- **neutral**: No particular emotion expressed