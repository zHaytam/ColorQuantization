using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;

namespace ColorQuantization
{
    public class Program
    {

        private static void Main(string[] args)
        {
            var inputFile = new FileInfo("test3.jpg");
            var outputFile = new FileInfo("result3.jpg");

            var mlContext = new MLContext();
            var img = LoadImage(inputFile);

            var fullData = mlContext.Data.LoadFromEnumerable(img.Data);
            var trainingData = mlContext.Data.LoadFromEnumerable(SelectRandom(img.Data, 1000));
            var model = Train(mlContext, trainingData, numberOfClusters: 32);

            VBuffer<float>[] centroidsBuffer = default;
            model.Model.GetClusterCentroids(ref centroidsBuffer, out int k);

            var labels = mlContext.Data
                .CreateEnumerable<Prediction>(model.Transform(fullData), reuseRowObject: false)
                .ToArray();

            Console.WriteLine("Reconstructing image...");
            using var reconstructedImg = ReconstructImage(labels, centroidsBuffer, img.Width, img.Height);
            SaveImage(reconstructedImg, outputFile);

            Console.WriteLine("Original size: {0:F2} KB.", inputFile.Length / 1024.0);
            Console.WriteLine("Result size: {0:F2} KB.", outputFile.Length / 1024.0);
        }

        private static void SaveImage(Image<Rgba32> image, FileInfo file)
        {
            using (var fs = new FileStream(file.FullName, FileMode.Create, FileAccess.Write))
            {
                image.SaveAsJpeg(fs);
            }
        }

        private static Image<Rgba32> ReconstructImage(Prediction[] labels, VBuffer<float>[] centroidsBuffer, int width, int height)
        {
            var img = new Image<Rgba32>(null, width, height);
            int i = 0;
            for (var h = 0; h < height; h++)
            {
                for (var w = 0; w < width; w++)
                {
                    var label = labels[i].PredictedLabel;
                    var centroid = centroidsBuffer[label - 1].DenseValues().ToArray();
                    img[w, h] = new Rgba32(centroid[0], centroid[1], centroid[2]);
                    i++;
                }
            }

            return img;
        }

        private static ClusteringPredictionTransformer<KMeansModelParameters> Train(MLContext mlContext, IDataView data, int numberOfClusters)
        {
            var pipeline = mlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);

            Console.WriteLine("Training model...");
            var sw = Stopwatch.StartNew();
            var model = pipeline.Fit(data);
            Console.WriteLine("Model trained in {0} ms.", sw.Elapsed.Milliseconds);

            return model;
        }

        private static ImageEntry LoadImage(FileInfo file)
        {
            using (Image<Rgba32> img = Image.Load<Rgba32>(file.FullName))
            {
                var pixels = new PixelEntry[img.Width * img.Height];

                int i = 0;
                foreach (var pixel in img.GetPixelSpan())
                {
                    pixels[i++] = new PixelEntry
                    {
                        Features = new[] 
                        {
                            (float)pixel.R / 255.0f,
                            (float)pixel.G / 255.0f,
                            (float)pixel.B / 255.0f
                        }
                    };
                }

                return new ImageEntry
                {
                    Data = pixels,
                    Width = img.Width,
                    Height = img.Height
                };
            }
        }

        private static T[] SelectRandom<T>(T[] array, int count)
        {
            var result = new T[count];
            var rnd = new Random();
            var chosen = new HashSet<int>();

            for (var i = 0; i < count; i++)
            {
                int r;
                while (chosen.Contains((r = rnd.Next(0, array.Length))))
                {
                    continue;
                }

                result[i] = array[r];
            }

            return result;
        }

    }

    public class PixelEntry
    {

        [VectorType(3)]
        public float[] Features { get; set; }

    }

    public class ImageEntry
    {

        public PixelEntry[] Data { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }

    }

    public class Prediction
    {

        public uint PredictedLabel { get; set; }

    }
}
