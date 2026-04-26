package agents.anac.y2013.MetaAgent;

import java.util.HashMap;

import agents.anac.y2013.MetaAgent.agentsData.AgentData;
import agents.anac.y2013.MetaAgent.parser.cart.TreeParser;
import agents.anac.y2013.MetaAgent.parser.cart.tree.Node;
public class Parser {

	/**
	 * @param args
	 */
	public enum Type{
		CART,CARTCLASS,LINREG,LOGREG ,NN,NNCLASS
	}
	
	public static double getMean(AgentData agent,HashMap<String, Double> values){
		TreeParser p=new TreeParser(agent);
		Node n=p.Parse();
//		System.out.println(n);
		Node ans=n.getBestNode(values);
		return ans.get_mean();
	}

}