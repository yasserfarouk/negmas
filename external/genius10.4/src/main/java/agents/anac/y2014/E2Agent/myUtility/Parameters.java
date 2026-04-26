package agents.anac.y2014.E2Agent.myUtility;

public class Parameters {
	public double utility;
	public double time;
	public double alpha;
	public double beta;
	public double g;
	
	public Parameters(double u, double t, double a, double b, double gValue) {
		utility = u;
		time = t;
		alpha = a;
		beta = b;
		g = gValue;
	}
	
	public String toString() {
		return "U: " + utility + " T: " + time + " a: " + alpha + " b: " + beta + " g: " + g;
	}
}
