package genius.core.logging;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Statistic results from one agent. immutable.
 *
 */
@XmlRootElement
public class AgentStatistics {
	@XmlElement
	private String agentname = null;

	@XmlElement
	private double totalUndiscountedUtility = 0;
	@XmlElement
	private double totalDiscountedUtility = 0;
	@XmlElement
	private long numberOfSessions = 0;
	@XmlElement
	private double totalNashDist;
	@XmlElement
	private double totalWelfare;
	@XmlElement
	private double totalParetoDistance;

	/**
	 * For de-serialization
	 */
	private AgentStatistics() {
	}

	/**
	 * @param name
	 *            the agent's name
	 * @param undiscounted
	 *            the total undiscounted utility accumulated over nSessions
	 * @param discounted
	 *            the total discounted utility accumulated over nSessions
	 * @param nash
	 *            the total distance to nash accumulated over nSessions
	 * @param welfare
	 *            the total social welfare accumulated over nSessions
	 * @param paretodist
	 *            the total accumulated distance to paretofrontier over
	 *            nSessions
	 * @param nSessions
	 *            the total number of sessions that was accumulated data over.
	 */
	public AgentStatistics(String name, double undiscounted, double discounted, double nash, double welfare,
			double paretodist, long nSessions) {
		agentname = name;
		totalUndiscountedUtility = undiscounted;
		totalDiscountedUtility = discounted;
		totalNashDist = nash;
		totalWelfare = welfare;
		totalParetoDistance = paretodist;
		numberOfSessions = nSessions;
	}

	/**
	 * Adds a new utility and returns a new statistic object with that.
	 * 
	 * @param undiscounted
	 *            the un-discounted utility to be added
	 * @param discounted
	 *            the discounted utility to be added
	 * @param nashdist
	 *            the nash distance to be added
	 * @param welfare
	 *            the social welfare to be added
	 * @param paretodist
	 *            the distance to pareto to be added
	 */
	public AgentStatistics withUtility(double undiscounted, double discounted, double nashdist, double welfare,
			double paretodist) {
		return new AgentStatistics(agentname, totalUndiscountedUtility + undiscounted,
				totalDiscountedUtility + discounted, totalNashDist + nashdist, totalWelfare + welfare,
				totalParetoDistance + paretodist, numberOfSessions + 1);
	}

	public String toString() {
		return "statistic of " + agentname + ":" + getMeanUndiscounted() + " " + getMeanDiscounted();
	}

	/**
	 * 
	 * @return the mean discounted utility of the agent
	 */
	@XmlElement // this puts the mean get into the serialized objects
	public double getMeanDiscounted() {
		return numberOfSessions == 0 ? 0 : totalDiscountedUtility / numberOfSessions;
	}

	/**
	 * 
	 * @return the mean undiscounted utility of the agent
	 */
	@XmlElement // this puts the mean get into the serialized objects
	public double getMeanUndiscounted() {
		return numberOfSessions == 0 ? 0 : totalUndiscountedUtility / numberOfSessions;
	}

	/**
	 * 
	 * @return name of agent for which the statistics are stored.
	 */
	public String getName() {
		return agentname;
	}

	/**
	 * 
	 * @return the mean nash distance of the agent
	 */
	@XmlElement // this puts the mean get into the serialized objects
	public double getMeanNashDistance() {
		return numberOfSessions == 0 ? 0 : totalNashDist / numberOfSessions;
	}

	/**
	 * 
	 * @return the mean social welfare of the agent
	 */
	@XmlElement // this puts the mean get into the serialized objects
	public double getMeanWelfare() {
		return numberOfSessions == 0 ? 0 : totalWelfare / numberOfSessions;
	}

	/**
	 * 
	 * @return the mean distance to pareto of the agent
	 */
	@XmlElement // this puts the mean get into the serialized objects
	public double getMeanParetoDistance() {
		return numberOfSessions == 0 ? 0 : totalParetoDistance / numberOfSessions;
	}
}
