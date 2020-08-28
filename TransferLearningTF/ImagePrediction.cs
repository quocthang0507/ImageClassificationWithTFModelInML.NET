using System;
using System.Collections.Generic;
using System.Text;

namespace TransferLearningTF
{
	public class ImagePrediction : ImageData
	{
		/// <summary>
		/// Score contains the confidence percentage for a given image classification
		/// </summary>
		public float[] Score;

		/// <summary>
		/// PredictedLabelValue contains a value for the predicted image classification label
		/// </summary>
		public string PredictedLabelValue;
	}
}
