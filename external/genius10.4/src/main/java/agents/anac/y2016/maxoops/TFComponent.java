/*
 * Author: Max W. Y. Lam (Aug 1 2015)
 * Version: Milestone 1
 * 
 * */

package agents.anac.y2016.maxoops;

import java.util.ArrayList;
import java.util.Random;

import genius.core.timeline.TimeLineInfo;

public class TFComponent {

	// Adjustable Parameters
	protected final int L;
	protected double alpha, beta, tau, zeta, lambda, gamma;

	// Fixed Parameters
	public final static String trendType = "linear damped",
			costType = "linear";

	private MaxOops agent;
	private TimeLineInfo timeline;
	public double[] adapSubInvMean, countSubInv;
	public double[][] oppAdapSubInvMean, oppCountSubInv;
	public int l;
	public double lltime, dltime, competitiveness;
	public double compaction, lf_slope, lt_slope, lt_filp, lconcess, lt_f;
	Random rand;

	// For Debug
	public ArrayList<Double> f_slope_time_series = null;
	public ArrayList<Double> t_filp_time_series = null;
	public ArrayList<Double> concess_time_series = null;

	public TFComponent(MaxOops agent, TimeLineInfo timeline) {
		// Set Adjustable Parameters First
		agent.params.addParam("TFComponent.L",
				(int) (30. / Math.pow(agent.delta, 0.35)));
		this.L = (int) agent.params.getParam("TFComponent.L");
		agent.params.addParam("TFComponent.alpha", 0);
		this.alpha = agent.params.getParam("TFComponent.alpha");
		agent.params.addParam("TFComponent.beta", 1.5);
		this.beta = agent.params.getParam("TFComponent.beta");
		agent.params.addParam("TFComponent.tau", 0.75);
		this.tau = agent.params.getParam("TFComponent.tau");
		agent.params.addParam("TFComponent.zeta", 1.5);
		this.zeta = agent.params.getParam("TFComponent.zeta");
		agent.params.addParam("TFComponent.lambda", 0.0);
		this.lambda = agent.params.getParam("TFComponent.lambda");
		agent.params.addParam("TFComponent.gamma", 0.5);
		this.gamma = agent.params.getParam("TFComponent.gamma");

		this.rand = new Random(agent.hashCode());
		this.agent = agent;
		this.l = 0;
		this.competitiveness = 0;
		this.lltime = 0;
		this.dltime = 0;
		this.compaction = agent.maxUtil - agent.minUtil;
		this.timeline = timeline;
		this.lf_slope = agent.secMaxUtil * (1 - agent.delta) + agent.maxUtil
				* agent.delta;
		this.lt_slope = 0;
		this.lt_f = 0;
		this.lt_filp = agent.delta * 0.85;
		this.lconcess = 0;
		this.adapSubInvMean = new double[L];
		this.countSubInv = new double[L];
		this.oppAdapSubInvMean = new double[agent.numParties - 1][L];
		this.oppCountSubInv = new double[agent.numParties - 1][L];
		for (int i = 0; i < L; i++) {
			adapSubInvMean[i] = 0;
			countSubInv[i] = 0;
			for (int j = 0; j < agent.numParties - 1; j++) {
				oppAdapSubInvMean[j][i] = 0;
				oppCountSubInv[j][i] = 0;
			}
		}
		this.f_slope_time_series = null;
		this.t_filp_time_series = null;
		this.concess_time_series = null;
	}

	public void setCompaction(double val) {
		compaction = val;
	}

	public double opponentsConcessionSlope() {
		try {
			double maxConcessionSlpoe = agent.delta - tau - 2;
			double minConcessionSlpoe = agent.delta - tau + 1;
			for (int i = 0; i < agent.numParties - 1; i++) {
				double slope = oppAdapSubInvMean[i][l]
						- oppAdapSubInvMean[i][Math.max(0, l - 1)];
				slope /= Math.max(dltime, 0.001);
				// maximum of slope = minimum of concession rate
				maxConcessionSlpoe = Math.min(
						Math.max(maxConcessionSlpoe, slope), 1.);
				minConcessionSlpoe = Math.max(
						Math.min(minConcessionSlpoe, slope), -1.);
			}
			double concess = (maxConcessionSlpoe * (1. - agent.delta) + minConcessionSlpoe
					* agent.delta);
			if (Double.isNaN(concess)) {
				concess = 0;
			}
			return concess;
		} catch (Exception e) {
			return 0;
		}
	}

	public int getSubIntvByTime(double time) {
		int i;
		for (i = 0; i < L; i++) {
			if (time < (i + 1) * 1. / L) {
				break;
			}
		}
		return i;
	}

	public int getCurrentSubIntv() {
		return getSubIntvByTime(timeline.getTime());
	}

	public void recordUtility(double util, int opponent) {
		double time = timeline.getTime();
		countSubInv[l] += 1.;
		if (opponent >= 0) {
			oppCountSubInv[opponent][l] += 1.;
		}
		if (l < 2) {
			adapSubInvMean[l] += (util - adapSubInvMean[l]) / countSubInv[l];
			if (opponent >= 0) {
				oppAdapSubInvMean[opponent][l] += (util - oppAdapSubInvMean[opponent][l])
						/ countSubInv[l];
			}
		} else {
			adapSubInvMean[l] += (util - (0.6 * adapSubInvMean[l] + 0.3
					* adapSubInvMean[l - 1] + 0.1 * adapSubInvMean[l - 2]))
					/ countSubInv[l];
			if (opponent >= 0) {
				oppAdapSubInvMean[opponent][l] += (util - (0.6
						* oppAdapSubInvMean[opponent][l] + 0.3
						* oppAdapSubInvMean[opponent][l - 1] + 0.1 * oppAdapSubInvMean[opponent][l - 2]))
						/ oppCountSubInv[opponent][l];
			}
		}
		if (l < getCurrentSubIntv()) {
			filppingTime();
			dltime = time - lltime;
			lltime = time;
			double u_low = concessionLimit();
			double concess = opponentsConcessionSlope();
			competitiveness = (agent.medianUtil / Math.pow(agent.delta, time) - adapSubInvMean[l])
					* 0.6 + competitiveness * 0.3;
			double f_slope = lf_slope;
			if (time < 0.2 - 0.4 * agent.delta) {
				f_slope *= Math.pow(agent.delta, 0.2 * time);
			}
			if (competitiveness > alpha) {
				lambda = -(lf_slope - adapSubInvMean[l])
						/ Math.sqrt(agent.delta);
				beta *= 0.98;
				competitiveness -= time * agent.stdUtil * agent.delta;
			} else {
				lambda = (1. - 0.5 / Math.sqrt(agent.delta)) * agent.stdUtil;
				competitiveness += time / agent.stdUtil * agent.delta
						* Math.sqrt(1 - time);
			}
			if (adapSubInvMean[l] >= agent.uqUtil) {
				double state1 = (adapSubInvMean[l] - agent.uqUtil)
						* (time - lt_slope) / agent.stdUtil;
				MaxOops.log1.println("State 1: " + state1);
				f_slope += state1;
			} else if (adapSubInvMean[l] >= agent.medianUtil) {
				double state2 = (agent.uqUtil - f_slope) * (time - lt_slope)
						/ agent.stdUtil;
				MaxOops.log1.println("State 2: " + state2);
				f_slope += state2;
			} else {
				double state3 = (adapSubInvMean[l] - f_slope)
						* (time - lt_slope) / agent.stdUtil;
				MaxOops.log1.println("State 3: " + state3);
				f_slope += state3;
			}
			f_slope += gamma
					* ((concess + lambda) - adapSubInvMean[l] + u_low
							+ Math.pow(agent.delta, 2)
							/ (1 + zeta + agent.delta) - (lf_slope - agent.theta)
							* Math.pow(time, 3 * agent.delta))
					* (time - lt_slope);
			if (time > lt_filp) {
				double u_epl = (adapSubInvMean[l] + beta * agent.stdUtil)
						* Math.sqrt(1.2 - agent.delta)
						+ (agent.meanUtil + beta * agent.stdUtil)
						* (1 - Math.sqrt(1.2 - agent.delta));
				f_slope += (f_slope - u_epl) / (time - 1.) * (time - lt_slope);
				if (concess > 0) {
					f_slope += (time - lt_slope) * (1 - time) * concess
							* agent.delta;
				}
			} else {
				if (competitiveness > 0) {
					f_slope -= (time - lt_slope) * (1 - time) / agent.stdUtil;
				}
				if (concess > 0) {
					f_slope += (time - lt_slope) * (1 - time) * concess
							/ agent.stdUtil;
				}
			}
			f_slope = Math.max(Math.max(u_low, f_slope), adapSubInvMean[l]);
			MaxOops.log2.println(competitiveness + ", " + alpha + ", " + beta
					+ ", " + zeta + ", " + lambda + ", " + tau + ", " + gamma
					+ ", " + f_slope);
			lf_slope = f_slope;
			lconcess = concess;
			lt_slope = time;
			if (agent.DEBUG && time > 0.98) {
				// PLOTComponent.plotDataPoints(f_slope_time_series,
				// "f_slope Time Series at l="+String.valueOf(l+1));
				// PLOTComponent.plotDataPoints(t_filp_time_series,
				// "t_filp Time Series at l="+String.valueOf(l+1));
				// PLOTComponent.plotDataPoints(concess_time_series,
				// "concess Time Series at l="+String.valueOf(l+1));
				// agent.DEBUG = false;
			}
			l++;
		}
	}

	public double filppingTime() {
		double time = timeline.getTime();
		double concess = opponentsConcessionSlope();
		double t_filp = lt_filp - Math.pow(1. - agent.delta - lt_filp, 2.)
				* concess * (time - lt_f);
		lt_f = time;
		lt_filp = t_filp;
		return Math.max(Math.min(t_filp, agent.delta * 1.5), agent.delta / 2.);
	}

	public double concessionLimit() {
		double u_least = Math.max(agent.minUtil + agent.delta * agent.stdUtil,
				agent.theta);
		double u_mu = agent.meanUtil;
		double u_med = agent.medianUtil;
		double u_low = Math
				.max(Math.max(Math.min(u_mu, u_med), adapSubInvMean[l]
						* agent.delta)
						- (0.5 - agent.delta) * agent.stdUtil, u_least);
		return u_low;
	}

	public double thresholdFunc() {
		// double time = timeline.getTime();
		// MaxOops.log1.println("time = "+time+", f_slope = "+lf_slope+", median = "+agent.medianUtil+", competitiveness = "+competitiveness
		// +", mean = "+agent.meanUtil+", adap mean = "+adapSubInvMean[l]+", std = "+agent.stdUtil
		// +", lq = "+agent.lqUtil+", uq = "+agent.uqUtil+", concess = "+lconcess
		// +", ulow = "+concessionLimit()+", t_filp = "+lt_filp);
		if (f_slope_time_series == null) {
			f_slope_time_series = new ArrayList<Double>();
			t_filp_time_series = new ArrayList<Double>();
			concess_time_series = new ArrayList<Double>();
		}
		f_slope_time_series.add(lf_slope);
		t_filp_time_series.add(0, lt_filp);
		concess_time_series.add(0, lconcess);
		return lf_slope;
	}

}
