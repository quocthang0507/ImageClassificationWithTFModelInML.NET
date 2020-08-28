using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TransferLearningTF
{
	class Program
	{
		static readonly string _assetsPath = Path.Combine(Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..\\..\\..\\")), "assets");
		static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
		static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
		static readonly string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
		static readonly string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
		static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

		static void Main(string[] args)
		{
			// MLContext class is a starting point for all ML.NET operations
			// It's similar, conceptuallu, to DBContext in Entity Framework
			MLContext mlContext = new MLContext();
			ITransformer model = GenerateModel(mlContext);
			ClassifySingleImage(mlContext, model);
			Console.ReadKey();
		}

		/// <summary>
		/// Map the parameter values to friendly names
		/// </summary>
		private struct InceptionSettings
		{
			public const int ImageHeight = 224;
			public const int ImageWidth = 224;
			public const float Mean = 117;
			public const float Scale = 1;
			public const bool ChannelsLast = true;
		}

		/// <summary>
		/// Contruct the ML.NET model pipeline
		/// An ML.NET model pipeline is a chain of estimators, trains the pipeline to produce the ML.NET model.
		/// It also evaluates the model against some previously unseen test data.
		/// </summary>
		/// <param name="mlContext"></param>
		/// <returns></returns>
		public static ITransformer GenerateModel(MLContext mlContext)
		{
			IEstimator<ITransformer> pipeline =
				// Add the estimators to load, resize and extract the pixels from the image data into a numeric vector
				// The image data needs to be processed into the format that the TF model expects
				mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
				// The image transforms transform the images into the model's expected format.
				.Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
				.Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
				// Add the estimator to load the TF model, and score it
				// It loads the TF model into memory, then processes the vector of pixel values through the TF model network.
				// Applying inputs to a deep learning model, and generating an output using the model, is referred to as Scoring
				.Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
				// Add the estimator to map the string labels in the training data to integer key values
				// The ML.NET trainer that is appended next requires its labels to be in key format rather than arbitrary strings
				.Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
				// Add the ML.NET training algorithm
				.Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
				// Add the estimator to map the predicted key value back into a string
				.Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel")).AppendCacheCheckpoint(mlContext);
			// IDataView is a flexible, effcient way of describing tabular data (numeric and text)
			IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);
			// Trains the model by applying the training dataset to the pipeline
			ITransformer model = pipeline.Fit(trainingData);
			// Load and transform the test data, by adding the following code to the next line of the GenerateModel method
			// It is able to use few sample images to evaluate the model
			IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
			IDataView predictions = model.Transform(testData);
			// Create an IEnumerable for the predictions for displaying results
			IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
			DisplayResults(imagePredictionData);
			// The Evaluate() method assesses the model, compares the predicted valuess with the test dataset labels
			// Returns the model performance metrics
			MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");
			// Display the metrics, share the results, and then act on them
			// The Log-loss should be as close to zero as possible
			Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
			Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
			// Return the trained model
			return model;
		}

		/// <summary>
		/// Perform a prediction on a single instance of data
		/// It's acceptable to use in single-threaded or prototype environments.
		/// For improved performance and thread safety in production environments, use the PredictionEnginePool servie
		/// </summary>
		/// <param name="mlContext"></param>
		/// <param name="model"></param>
		public static void ClassifySingleImage(MLContext mlContext, ITransformer model)
		{
			var imageData = new ImageData() { ImagePath = _predictSingleImage };
			var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
			var prediction = predictor.Predict(imageData);
			Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
		}

		/// <summary>
		/// Handle displaying the image and prediction results
		/// </summary>
		/// <param name="imagePredictionData"></param>
		private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
		{
			foreach (ImagePrediction prediction in imagePredictionData)
			{
				Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
			}
		}

		/// <summary>
		/// Parse through the tags.tsv file to add the file path to the image file name for the ImagePath property and load it and the Label into an ImageData object
		/// </summary>
		/// <param name="file"></param>
		/// <param name="folder"></param>
		/// <returns></returns>
		public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
		{
			return File.ReadAllLines(file).Select(line => line.Split('\t')).Select(line => new ImageData()
			{
				ImagePath = Path.Combine(folder, line[0])
			});
		}
	}
}
