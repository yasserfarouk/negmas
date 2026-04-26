package negotiator.boaframework.offeringstrategy.anac2010.IAMhaggler2010;

public class TimeConcessionFunction extends ConcessionFunction {

	private double beta;
	private double breakoff;

	public TimeConcessionFunction(double beta, double breakoff) {
		this.beta = beta;
		this.breakoff = breakoff;
	}

	@Override
	public double getConcession(double startUtility, long currentTime, double totalTime) {
		return startUtility - (startUtility - breakoff) * Math.pow(currentTime / totalTime, 1.0 / beta);
	}

	public class Beta {
		public static final double CONCEDER_EXTREME = 5.0;
		public static final double CONCEDER = 2.0;
		public static final double LINEAR = 1.0;
		public static final double BOULWARE = 0.5;
		public static final double BOULWARE_EXTREME = 0.2;
	}

	public static final double BREAKOFF = 0.0;
	public static final double DEFAULT_BREAKOFF = 0.5;
}
