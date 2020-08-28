using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TransferLearningTF
{
	public class ImageData
	{
		/// <summary>
		/// ImagePath contains the image file name
		/// </summary>
		[LoadColumn(0)]
		public string ImagePath;

		/// <summary>
		/// Label contains a value for the image label
		/// </summary>
		[LoadColumn(1)]
		public string Label;
	}
}
