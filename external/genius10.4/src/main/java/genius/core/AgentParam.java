package genius.core;
/**
 * This class stores info about a parameter of an agent.
 *  @author W.Pasman 19aug08
 */

public class AgentParam {
	public String agentclass; 	// the agent class for which this is a parameter.
								// we do not refer to Class because that suggests loading its definition etc
								// and that may not be possible, especially from static contexts.
	public String name;
	public Double min;
	public Double max;
	
	public AgentParam(String agentclassP, String nameP, Double minP, Double maxP)
	{
		agentclass=agentclassP;
		name=nameP;
		min=minP;
		max=maxP;
	}
	
	static final long serialVersionUID=0;
	
	public boolean equals(Object o) {
		if (!(o instanceof AgentParam)) return false;
		AgentParam ap=(AgentParam)o;
		return ap.agentclass.equals(agentclass) && ap.name.equals(name) /*&& ap.min==min && ap.max==max*/;
	}
	public String toString() {
		return agentclass+":"+name;
	}
	
}