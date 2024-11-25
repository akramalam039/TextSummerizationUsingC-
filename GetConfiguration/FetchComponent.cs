using Microsoft.ML;

namespace GetConfiguration
{
    public class SearchData
    {
        public string ComponentName { get; set; }
        public string ComponentType { get; set; }
        public string SubComponentType { get; set; }
    }

    public class SearchQuery
    {
        public string Query { get; set; }
    }
    public class FeaturePrediction
    {
        public float[] Features { get; set; }
    }
    internal class SimpleSearch
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        public SimpleSearch(MLContext mlContext, ITransformer model)
        {
            _mlContext = mlContext;
            _model = model;
        }

        public SearchData SearchBestMatch(string userQuery, List<SearchData> data)
        {
            var queryData = new List<SearchData> { new() { ComponentName = userQuery, SubComponentType = userQuery, ComponentType = userQuery } };
            var queryDataView = _mlContext.Data.LoadFromEnumerable(queryData);

            // Preprocess the query (featurize it)
            var queryTransformed = _model.Transform(queryDataView);

            // Now calculate cosine similarity between the query and data entries
            var dataView = _mlContext.Data.LoadFromEnumerable(data);
            var transformedData = _model.Transform(dataView);

            var queryFeatures = _mlContext.Data.CreateEnumerable<FeaturePrediction>(queryTransformed, reuseRowObject: false).First();
            var dataFeatures = _mlContext.Data.CreateEnumerable<FeaturePrediction>(transformedData, reuseRowObject: false).ToList();

            var bestMatch = dataFeatures
                .Select((dataEntry, index) => new
                {
                    Data = data[index],
                    Score = CalculateCosineSimilarity(queryFeatures.Features, dataEntry.Features)
                })
                .OrderByDescending(x => x.Score)
                .FirstOrDefault();

            var bestMatchList = dataFeatures
                .Select((dataEntry, index) => new
                {
                    Data = data[index],
                    Score = CalculateCosineSimilarity(queryFeatures.Features, dataEntry.Features)
                })
                .OrderByDescending(x => x.Score)
                .ToList();
            foreach (var dataEntry in bestMatchList.Where(x => x.Score > Math.Round(0.50f)))
            {
                Console.WriteLine(string.Join(" - ", dataEntry.Data.ComponentName, dataEntry.Data.ComponentType, dataEntry.Data.SubComponentType));
            }
            return bestMatch?.Data;
        }

        // Calculate cosine similarity between two feature vectors
        private double CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            var dotProduct = vector1.Zip(vector2, (x, y) => x * y).Sum();
            var magnitude1 = Math.Sqrt(vector1.Sum(x => x * x));
            var magnitude2 = Math.Sqrt(vector2.Sum(x => x * x));

            return dotProduct / (magnitude1 * magnitude2);
        }
    }

    internal class FetchComponent
    {
    }
}
