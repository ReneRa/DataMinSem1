package cifo;

import java.awt.image.BufferedImage;
import java.awt.image.PixelGrabber;

import gd.gui.GeneticDrawingView;

public class ProblemInstance {

	public static BufferedImage currentImage;
	public static GeneticDrawingView view;

	protected BufferedImage targetImage;
	protected int[] targetPixels;
	protected int numberOfTriangles;

	public ProblemInstance(int numberOfTriangles) {
		targetImage = currentImage;
		computeTargetPixels();
		this.numberOfTriangles = numberOfTriangles;
	}

	protected void computeTargetPixels() {
		targetPixels = new int[targetImage.getWidth() * targetImage.getHeight()];
		PixelGrabber pg = new PixelGrabber(targetImage, 0, 0, targetImage.getWidth(), targetImage.getHeight(),
				targetPixels, 0, targetImage.getWidth());
		try {
			pg.grabPixels();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	public BufferedImage getTargetImage() {
		return targetImage;
	}

	public int getImageWidth() {
		return targetImage.getWidth();
	}

	public int getImageHeight() {
		return targetImage.getHeight();
	}

	public int[] getTargetPixels() {
		return targetPixels;
	}

	public int getNumberOfTriangles() {
		return numberOfTriangles;
	}
}
