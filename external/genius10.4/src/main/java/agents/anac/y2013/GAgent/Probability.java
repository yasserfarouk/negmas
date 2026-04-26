package agents.anac.y2013.GAgent;

public class Probability {
	
	public double mean;
	public double widthUtil;
	public double variance;
	public int itemNum;
	
	private double sumRatio;
	private double sumRatio2;
	
	
	public Probability(double w) {
		sumRatio = 0;
		sumRatio2 = 0;
		mean = 0D;
		widthUtil = w;
		variance = 0D;
		itemNum = 0;
	}
	private void calMean(double ratio){
		sumRatio += ratio;
		mean = sumRatio / itemNum;
	}
	
	private double calVar(double ratio){
		sumRatio2 += ratio * ratio; 
		variance = sumRatio2 / itemNum - mean*mean;
		return variance;
	}
	
	private double getRatio(double util){
		return  util / widthUtil;
	}
	
	public double getM(double diff){
		itemNum++;
		double ra = getRatio(diff);
		calMean(ra);
		return mean;
	}
	public double getVar(double diff){
		itemNum++;
		double ra = getRatio(diff);
		calMean(diff);
		variance = calVar(ra);
		return mean*100000*variance*widthUtil;
	}
	

}
