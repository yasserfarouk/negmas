package agents.anac.y2010.Southampton.utils.concession;

import java.util.ArrayList;
import agents.anac.y2010.Southampton.utils.Pair;

public class ConcessionUtils {

	public static double getBeta(ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory, double time, double discounting, double utility0,
			double utility1) {
		return getBeta(bestOpponentBidUtilityHistory, time, discounting, utility0, utility1, 0.1, 0.01, 2, 1, 1, 1);
	}
	
	public static double getBeta(ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory, double time, double discounting, double utility0,
			double utility1, double minDiscounting, double minBeta, double maxBeta, double defaultBeta, double ourTime, double opponentTime) {
		discounting = Math.max(discounting, minDiscounting);
		try {
			// Estimate the alpha value.
			//double alpha = getAlpha(bestOpponentBidUtilityHistory);
			Pair<Double, Double> params = getRegressionParameters(bestOpponentBidUtilityHistory, utility0);
			double a = params.fst;
			double b = params.snd;
			// Find the maxima.
			double maxima = getMaxima(a, b, time, discounting, utility0, opponentTime);
			// Find the current utility of our current concession strategy.
			//double util = utility0 + (utility1 - utility0) * (1 - Math.exp(-alpha * maxima));
			double util = utility0 + (Math.exp(a) * Math.pow(maxima, b));
			util = Math.max(0, Math.min(1, util));
			// Calculate the beta value.
			double beta = Math.max(0, Math.log(maxima) / Math.log((1 - util) / 0.5));
			/*
			if (time < 0.5) {
				double weightBeta = time / 0.5;
				//beta = (beta * weightBeta) + (defaultBeta * (1 - weightBeta));
				beta = Math.exp(Math.log(beta) * weightBeta) + Math.exp(Math.log(defaultBeta) * (1 - weightBeta));
			}
			*/

			return Math.min(maxBeta, Math.max(minBeta, beta));
		} catch (Exception ex) {
			return defaultBeta;
		}
	}

	private static Pair<Double, Double> getRegressionParameters(ArrayList<Pair<Double, Double>> bestOpponentBidUtilityHistory, double utility0) {
		double n = 1;
		double x = 0;//Math.log(1);
		double y = Math.log(1 - utility0);
		double sumlnxlny = 0;//x * y;
		double sumlnx = 0;//x;
		double sumlny = y;
		double sumlnxlnx = 0;//x * y;
		for (Pair<Double, Double> d : bestOpponentBidUtilityHistory) {
			x = Math.log(d.snd);
			y = Math.log(d.fst - utility0);

			if(Double.isNaN(x))
				throw new RuntimeException("Unable to perform regression using provided points (x).");
			if(Double.isNaN(y))
				throw new RuntimeException("Unable to perform regression using provided points (y).");
			if(Double.isInfinite(x) || Double.isInfinite(y))
				continue;
			
			sumlnxlny += x * y;
			sumlnx += x;
			sumlny += y;
			sumlnxlnx += x * x;
			n++;
		}
		
		double b = ((n * sumlnxlny) - (sumlnx * sumlny)) / ((n * sumlnxlnx) - (sumlnx * sumlnx));
		if(Double.isNaN(b))
			throw new RuntimeException("Unable to perform regression using provided points (b)." + (sumlnxlny) + ", " + (n * sumlnxlnx) + ", " + (sumlnx * sumlnx));
		double a = (sumlny - (b * sumlnx)) / n;

		if(Double.isNaN(a))
			throw new RuntimeException("Unable to perform regression using provided points (a).");
		return new Pair<Double, Double>(a, b);
	}

	/*
	private static double getMaxima(double alpha, double time, double discounting, double utility0, double utility1) {
		double alpharoot = Math.log((discounting * utility1) / ((alpha + discounting) * (utility1 - utility0))) / -alpha;
		ArrayList<Double> maxima = new ArrayList<Double>();
		if (alpharoot >= 0 && alpharoot <= 1 && alpharoot > time) {
			maxima.add(alpharoot);
		}
		maxima.add(time);
		maxima.add(1.0);
		double maxUtil = 0;
		double result = 0;
		for (double maximum : maxima) {
			double util = (utility0 + ((utility1 - utility0) * (1 - Math.exp(-alpha * maximum)))) * Math.exp(-discounting * maximum);
			if (util > maxUtil) {
				result = maximum;
				maxUtil = util;
			}
		}
		return result;
	}
	*/

	private static double getMaxima(double a, double b, double time, double discounting, double utility0, double opponentTime) {
		//double root = b / discounting;
		ArrayList<Double> maxima = new ArrayList<Double>();
		maxima.add(time);
		for(int i=(int) Math.floor(time*1000); i<=1000; i++) {
			double root = (double)i / 1000.0;
			//if (root >= 0 && root <= 1 && root > time) {
				maxima.add(root);
			//}
		}
		maxima.add(1.0);
		double maxUtil = 0;
		double result = 0;
		
		double timeScaledDiscounting = -discounting * getTimeScaleFactor(time, opponentTime);
		double expA = Math.exp(a);
		
		for (double maximum : maxima) {
			double util = (utility0 + (expA * Math.pow(maximum, b))) * Math.exp(timeScaledDiscounting * maximum);
			if (util > maxUtil) {
				result = maximum;
				maxUtil = util;
			}
		}
		return result;
	}

	private static double getTimeScaleFactor(double ourTime, double opponentTime) {
		double sf = (ourTime + opponentTime)/(ourTime * 2.0);
		return sf;
	}

}
