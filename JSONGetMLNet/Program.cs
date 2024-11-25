using Microsoft.ML;

namespace JSONGetMLNet
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            var mlContext = new MLContext();

            // Load data from JSON or define a collection
            var jsonData = new List<SearchData>
            {
                new SearchData { Id = 1, Title = "Machine Learning Basics", Description = "Introduction to machine learning and its applications." },
                new SearchData { Id = 2, Title = "Deep Learning", Description = "Learn about deep neural networks and advanced topics." },
                new SearchData { Id = 3, Title = "Data Science Overview", Description = "Overview of data science including tools and techniques." }
            };

            var dataView = mlContext.Data.LoadFromEnumerable(jsonData);
            var schema = dataView.Schema;
            foreach (var column in schema)
            {
                Console.WriteLine($"{column.Name}: {column.Type}");
            }
            // Create the text processing pipeline
            var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SearchData.Title));

            // Fit and transform the data
            var model = textPipeline.Fit(dataView);

            // Instantiate the search engine
            var searchEngine = new SimpleSearch(mlContext, model);

            // User input query
            var userQuery = "What is machine learning?";
            Console.WriteLine($"{nameof(userQuery)} : {userQuery} ");
            // Perform search and get the best match
            var bestMatch = searchEngine.SearchBestMatch(userQuery, jsonData);

            if (bestMatch != null)
            {
                Console.WriteLine($"Best Match: {bestMatch.Title} - {bestMatch.Description}- {bestMatch.Id}");
            }
            else
            {
                Console.WriteLine("No match found.");
            }

            Console.ReadLine();
        }
    }
}
