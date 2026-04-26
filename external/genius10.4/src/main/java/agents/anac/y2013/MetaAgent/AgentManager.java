package agents.anac.y2013.MetaAgent;

import java.io.Serializable;
import java.util.AbstractMap.SimpleEntry;
import java.util.HashMap;
import java.util.Set;

import agents.anac.y2013.MetaAgent.agentsData.AgentData;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataAgentLG;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataBRAMAgent2;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataCUHKAgent;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataIAMhaggler2012;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataOMACagent;
import agents.anac.y2013.MetaAgent.agentsData.agents.DataTheNegotiatorReloaded;
import agents.anac.y2013.MetaAgent.portfolio.AgentLG.*;
import agents.anac.y2013.MetaAgent.portfolio.AgentMR.*;
import agents.anac.y2013.MetaAgent.portfolio.BRAMAgent2.*;
import agents.anac.y2013.MetaAgent.portfolio.CUHKAgent.*;
import agents.anac.y2013.MetaAgent.portfolio.IAMhaggler2012.*;
import agents.anac.y2013.MetaAgent.portfolio.OMACagent.*;
import agents.anac.y2013.MetaAgent.portfolio.thenegotiatorreloaded.*;
import genius.core.*;


@SuppressWarnings("serial")
public class AgentManager implements Serializable{
	HashMap<String, SimpleEntry<Integer, SimpleEntry<Double, Double>>> agents; //<Agent name, <countUsages, predictedScore>>
	int playsCount = 0;
	int predictionFactor = 5;
	String selectedAgent = "";
	SimpleEntry<Integer, SimpleEntry<Double, Double>> selectedInfo = new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)); //<counter,predicted,previousScores>
	double averageUtility = 0;
	double stdev = 0;
	
	public AgentManager(){
		agents = new HashMap<String, SimpleEntry<Integer, SimpleEntry<Double, Double>>>();
		agents.put("AgentLG", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		//agents.put("AgentMR", new SimpleEntry<Integer, Double>(0, 0.0));
		agents.put("BRAMAgent2", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		agents.put("CUHKAgent", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		agents.put("IAMhaggler2012", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		agents.put("OMACagent", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		agents.put("TheNegotiatorReloaded", new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0,new SimpleEntry<Double, Double>(0.0,0.0)));
		
	}
	
	private Agent GetAgent(String name){
		
		if (name.equalsIgnoreCase("AgentLG")) return new AgentLG();
		else if (name.equalsIgnoreCase("AgentMR")) return new AgentMR();
		else if (name.equalsIgnoreCase("BRAMAgent2")) return new BRAMAgent2();
		else if (name.equalsIgnoreCase("CUHKAgent")) return new CUHKAgent();
		else if (name.equalsIgnoreCase("IAMhaggler2012")) return new IAMhaggler2012();
		else if (name.equalsIgnoreCase("OMACagent")) return new OMACagent();
		else if (name.equalsIgnoreCase("TheNegotiatorReloaded")) return new TheNegotiatorReloaded();
		
		return new CUHKAgent(); //default, as winner of 2012 contest.
	}
	
	public AgentData GetAgentData(String name){
		if (name.equalsIgnoreCase("AgentLG")) return new DataAgentLG();
		else if (name.equalsIgnoreCase("AgentMR")) return new DataAgentLG();
		else if (name.equalsIgnoreCase("BRAMAgent2")) return new DataBRAMAgent2();
		else if (name.equalsIgnoreCase("CUHKAgent")) return new DataCUHKAgent();
		else if (name.equalsIgnoreCase("IAMhaggler2012")) return new DataIAMhaggler2012();
		else if (name.equalsIgnoreCase("OMACagent")) return new DataOMACagent();
		else if (name.equalsIgnoreCase("TheNegotiatorReloaded")) return new DataTheNegotiatorReloaded();
		
		return new DataCUHKAgent(); //default, as winner of 2012 contest.
	}
	
	public Agent SelectBestAgent(){
		String bestAgent = "";
		double bestScore = -1;
		
		//String fileName = "MetaAgent_Log.csv"; //need to delete!
		//BufferedWriter out = null; //
		//try { //
		//	FileWriter fstream = new FileWriter(fileName,true); //
		//	out = new BufferedWriter(fstream); //
		
		for (String agent : agents.keySet()) {
			SimpleEntry<Integer, SimpleEntry<Double, Double>> information = agents.get(agent);
			double curr = (information.getValue().getValue()*information.getKey() + information.getValue().getKey()*predictionFactor)/(information.getKey()+predictionFactor) ;
			double curr2 = Math.sqrt(stdev * Math.log(playsCount+agents.size()*predictionFactor)/(information.getKey()+predictionFactor)); //UCB MAB: X_j + sqrt(2ln(n)/n_j)
			curr += curr2;
		//	out.append(agent + "," + information.getKey() + "," + information.getValue().getValue() + "," + information.getValue().getKey() +  "," + curr); //
		//	out.newLine();//
			
			if (bestScore < curr || //regular win
					(bestScore == curr && agents.get(bestAgent).getKey() > information.getKey())){ //tieBreak - take the agent with less performances
				bestScore = curr;
				bestAgent = agent;
				selectedInfo = information;
			}
		}
		selectedAgent = bestAgent;
		
		
		//System.out.println("selected Agent: " + bestAgent + " --> " + bestScore); //
		//out.append(",,,"+bestAgent+","+bestScore); //
		//out.newLine(); // 
			
		//} catch (Exception e) { System.out.println("Error in WriteFile: " + e.toString());		} //
		//finally{ try {  //
		//	out.close(); //
		//} catch (Exception e2) { }} //
		
		return GetAgent(selectedAgent);
	}
	
	public boolean IsUsed(){
		return playsCount > agents.size();
	}
	
	public Set<String> GetAgents(){
		return agents.keySet();
	}
	
	public void UpdateUtility (String agent, double util){
		if (agent == ""){
			selectedInfo = new SimpleEntry<Integer, SimpleEntry<Double, Double>>
				(selectedInfo.getKey()+1, new SimpleEntry<Double, Double>
				(selectedInfo.getValue().getKey(),
						(selectedInfo.getValue().getValue()*selectedInfo.getKey() + util - averageUtility)/(selectedInfo.getKey() + 1)));
			agents.put(selectedAgent, selectedInfo);
			//averageUtility = (averageUtility * playsCount + util) / (playsCount + 1);  //receiveMessage the average utility measure
			playsCount ++;
		}
		else
		{
			SimpleEntry<Integer, SimpleEntry<Double, Double>> info = new SimpleEntry<Integer, SimpleEntry<Double, Double>>(0, new SimpleEntry<Double, Double>(util, 0.0));
			agents.put(agent,info);
			//System.out.println(agent + " --> " + util); //delete
		}
		 
	}
	
	public void SetAvgUtil (double avg, double stdev){
		averageUtility = avg;
		this.stdev = stdev;
	}
}
