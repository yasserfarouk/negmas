package genius.core.logging;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

/**
 * Collects all statistics of all agents. immutable
 *
 */
@XmlRootElement
public class AgentsStatistics {
	@XmlElement
	private final List<AgentStatistics> statistics;

	/**
	 * Just for serializer
	 */
	@SuppressWarnings("unused")
	private AgentsStatistics() {
		statistics = null; // should be filled by deserializer
	}

	public AgentsStatistics(List<AgentStatistics> stats) {
		if (stats == null) {
			throw new NullPointerException("stats==null");
		}
		this.statistics = stats;
	}

	/**
	 * Adds or replaces statistic.
	 * 
	 * @param stats
	 * @return new AgentStatistics with the new Statistic added or replaced
	 */
	public AgentsStatistics withStats(AgentStatistics stats) {
		List<AgentStatistics> newlist = new ArrayList<AgentStatistics>(statistics);
		newlist.add(stats);
		return new AgentsStatistics(newlist);
	}

	/**
	 * See aso {@link #withStatistics(String, double, double)}
	 * 
	 * @param agentname
	 *            the agent for which statistics are needed.
	 * @return statistic with given agentname, or null if no such elemenet.
	 */
	public AgentStatistics get(String agentname) {
		int i = index(agentname);
		if (i == -1) {
			return null;
		}
		return statistics.get(i);
	}

	/**
	 * 
	 * @param agentname
	 * @return index of given agentname in the array, or -1
	 */
	private int index(String agentname) {
		for (int index = 0; index < statistics.size(); index++) {
			if (agentname.equals(statistics.get(index).getName())) {
				return index;
			}
		}
		return -1;

	}

	public List<AgentStatistics> getStatistics() {
		return Collections.unmodifiableList(statistics);
	}

	/**
	 * Update the statistic of given agent.
	 * 
	 * @param agent
	 * @param undiscounted
	 * @param discounted
	 * @param welfare
	 * @param nashdist
	 */
	public AgentsStatistics withStatistics(String agent, double undiscounted, double discounted, double nashdist,
			double welfare, double paretoDist) {
		AgentStatistics stat = get(agent);
		if (stat == null) {
			stat = new AgentStatistics(agent, undiscounted, discounted, nashdist, welfare, paretoDist, 1);
		} else {
			stat = stat.withUtility(undiscounted, discounted, nashdist, welfare, paretoDist);
		}
		return withStatistics(stat);
	}

	/**
	 * @param stat
	 * @return new AgentsStatistics with stat added/updated
	 */
	public AgentsStatistics withStatistics(AgentStatistics stat) {
		ArrayList<AgentStatistics> newstatistics = new ArrayList<>(statistics);
		int index = index(stat.getName());
		if (index == -1) {
			newstatistics.add(stat);
		} else {
			newstatistics.set(index, stat);
		}
		return new AgentsStatistics(newstatistics);
	}
}
