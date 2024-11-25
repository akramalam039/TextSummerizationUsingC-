using Microsoft.ML;

namespace GetConfiguration
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //To Do- https://dotnet.microsoft.com/en-us/apps/ai/ml-dotnet, https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/Clustering_CustomerSegmentation?WT.mc_id=dotnet-35129-website,
            //https://www.youtube.com/watch?v=C-lnYdAR9UI
            //https://github.com/dotnet/machinelearning/milestone/16
            Console.WriteLine("Hello, World!");

            var mlContext = new MLContext();

            // Load data from JSON or define a collection
            var jsonData = new List<SearchData>
            {
                new SearchData { ComponentName = "192.168.30.22", ComponentType = "File Server", SubComponentType = "Windows" },
                new SearchData { ComponentName = "192.168.30.220", ComponentType = "File Server", SubComponentType = "Nasuni" },
                new SearchData { ComponentName = "192.168.30.180", ComponentType = "File Server", SubComponentType = "Nutanix" },
                new SearchData { ComponentName = "192.168.30.200", ComponentType = "Active Directory", SubComponentType = "Active Directory" },
                new SearchData { ComponentName = "Devsoft", ComponentType = "SharePoint", SubComponentType = "SharePoint" },
                new SearchData { ComponentName = "LepSoft", ComponentType = "MS Team", SubComponentType = "MS Team" },
            };

            var dataView = mlContext.Data.LoadFromEnumerable(jsonData);
            var schema = dataView.Schema;
            foreach (var column in schema)
            {
                Console.WriteLine($"{column.Name}: {column.Type}");
            }
            // Create the text processing pipeline
            //var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SearchData.ComponentType));
            var textPipeline = mlContext.Transforms.Text.FeaturizeText("ComponentNameFeatures", nameof(SearchData.ComponentName))
                             .Append(mlContext.Transforms.Text.FeaturizeText("ComponentTypeFeatures", nameof(SearchData.ComponentType)))
                             .Append(mlContext.Transforms.Text.FeaturizeText("SubComponentTypeFeatures", nameof(SearchData.SubComponentType)))
                             .Append(mlContext.Transforms.Concatenate("Features", "ComponentNameFeatures", "ComponentTypeFeatures", "SubComponentTypeFeatures"));

            // Fit and transform the data
            var model = textPipeline.Fit(dataView);
            // Fit and transform the data
            //var model = textPipeline.Fit(dataView);

            // Instantiate the search engine
            var searchEngine = new SimpleSearch(mlContext, model);
            while (true)
            {
                var userQuery = Console.ReadLine() ?? "Show the file server report?";
                Console.WriteLine($"{nameof(userQuery)} : {userQuery} ");
                // Perform search and get the best match
                var bestMatch = searchEngine.SearchBestMatch(userQuery, jsonData);

                if (bestMatch != null)
                {
                    Console.WriteLine($"Best Match: {bestMatch.ComponentName} - {bestMatch.ComponentType} - {bestMatch.SubComponentType}");
                }
                else
                {
                    Console.WriteLine("No match found.");
                }
            }
            // User input query


            Console.ReadLine();


        }
    }
}
