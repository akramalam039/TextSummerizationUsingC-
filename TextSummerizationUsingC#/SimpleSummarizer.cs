using Microsoft.ML;
using Microsoft.ML.Data;

namespace TextSummerizationUsingC_
{
    internal class SimpleSummarizer
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        public SimpleSummarizer(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
        }

        public string SummarizeText(string inputText, int summaryLength)
        {
            // Step 1: Preprocess and tokenize text
            var preprocessedData = TextPreprocessor.PreprocessText(_mlContext, new[] { inputText });

            // Step 2: Use the trained model to predict sentence features (e.g., TF-IDF vectors)
            var predictions = _model.Transform(preprocessedData);

            // Step 3: Extract sentences from the input text
            var sentences = inputText.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);

            // Step 4: Score sentences based on TF-IDF scores (use a simple heuristic like summing feature values)
            var sentenceScores = sentences.Select((sentence, index) => new
            {
                Sentence = sentence,
                Score = GetSentenceScore(predictions, index)
            }).OrderByDescending(s => s.Score).Take(summaryLength);

            // Step 5: Combine top sentences to form a summary
            var summary = string.Join(" ", sentenceScores.Select(s => s.Sentence));
            return summary;
        }

        private float GetSentenceScore(IDataView predictions, int sentenceIndex)
        {
            // Implement a method to score the sentence, based on its TF-IDF or other features
            // This is a simplified example, you might need to extract features for each sentence and compute scores
            var featureColumn = predictions.GetColumn<float[]>("Features").ToArray();
            return featureColumn[0].Sum();
        }
    }
}
