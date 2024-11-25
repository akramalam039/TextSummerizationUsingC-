using Microsoft.ML;

namespace TextSummerizationUsingC_
{
    public class TextData
    {
        public string Text { get; set; }
    }

    public class SummarizedText
    {
        public string Summary { get; set; }
    }
    internal class TextPreprocessor
    {
        //public static IDataView PreprocessText(MLContext mlContext, string[] inputTexts)
        //{
        //    var data = inputTexts.Select(text => new TextData { Text = text }).ToArray();
        //    var dataView = mlContext.Data.LoadFromEnumerable(data);

        //    // Text transformation: Tokenize and clean text
        //    var textPipeline = mlContext.Transforms.Text
        //        .Concatenate("Features", nameof(TextData.Text))
        //        .Append(mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text)));

        //    var preprocessedData = textPipeline.Fit(dataView).Transform(dataView);
        //    return preprocessedData;
        //}

        public static IDataView PreprocessText(MLContext mlContext, string[] inputTexts)
        {
            // Convert input texts into an IDataView
            var data = inputTexts.Select(text => new TextData { Text = text }).ToArray();
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Tokenize and featurize text using TF-IDF
            var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(TextData.Text));
            var preprocessedData = textPipeline.Fit(dataView).Transform(dataView);

            return preprocessedData;
        }
    }
}
