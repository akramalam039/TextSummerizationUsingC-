using Microsoft.ML;

namespace TextSummerizationUsingC_
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            var mlContext = new MLContext();

            // Sample text
            string inputText = "Machine learning is an application of artificial intelligence. It gives computers the ability to learn without being explicitly programmed. Machine learning algorithms are used in various fields like finance, healthcare, and robotics.";

            // Preprocess text and build model
            var preprocessedData = TextPreprocessor.PreprocessText(mlContext, new[] { inputText });

            // Define a simple pipeline (for demonstration)
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text));
            var model = pipeline.Fit(preprocessedData);

            // Create the summarizer and generate a summary
            var summarizer = new SimpleSummarizer(mlContext, model);
            string summary = summarizer.SummarizeText(inputText, 2); // Get top 2 sentences

            Console.WriteLine("Summary: ");
            Console.WriteLine(summary);
        }
    }
}
