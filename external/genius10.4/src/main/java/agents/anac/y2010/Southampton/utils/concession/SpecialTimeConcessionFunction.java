package agents.anac.y2010.Southampton.utils.concession;

public class SpecialTimeConcessionFunction extends ConcessionFunction {

	private double beta;
	private double breakoff;
	private double defaultBeta;

	public SpecialTimeConcessionFunction(double beta, double defaultBeta, double breakoff) {
		this.beta = beta;
		this.breakoff = breakoff;
		this.defaultBeta = defaultBeta;
	}

	@Override
	public double getConcession(double startUtility, long currentTime, double totalTime) {
		double utilityBeta = startUtility - (startUtility - breakoff) * Math.pow(currentTime / totalTime, 1.0 / beta);
		double utilityDefaultBeta = startUtility - (startUtility - breakoff) * Math.pow(currentTime / totalTime, 1.0 / defaultBeta);
		double gamePercent = currentTime / totalTime;
		if(gamePercent < 0.2)
		{
			return utilityDefaultBeta;
		}
		if(gamePercent < 0.4)
		{
			return (utilityDefaultBeta * (1 - ((gamePercent - 0.2)/0.2))) + (utilityBeta * ((gamePercent - 0.2)/0.2));
		}
		return utilityBeta;
	}
}
