package agents.anac.y2013.MetaAgent.parser.cart;

import java.util.HashMap;

import agents.anac.y2013.MetaAgent.Parser.Type;
import agents.anac.y2013.MetaAgent.agentsData.AgentData;
import agents.anac.y2013.MetaAgent.parser.cart.tree.MeanNode;
import agents.anac.y2013.MetaAgent.parser.cart.tree.Node;

public class TreeParser {

	/**
	 * @param args
	 */
	String text="";
	Type type;
	AgentData data;
	public TreeParser(AgentData data){
		this.data=data;
	}
	public Node Parse(){
		text=data.getText();
		text=text.trim();
		String []nodesText=text.split("Node number ");
		return parseNodes(nodesText);
		
	}
	
	private Node parseNodes(String[] nodesText) {
		HashMap<Integer,Node> nodes= new HashMap<Integer,Node>();
		for (int i = 0; i < nodesText.length; i++) {
			if(!nodesText[i].equals("")){
				Node n=MeanNode.factory(nodesText[i]);
				nodes.put(new Integer(n.get_id()),n);
			}	
		}
			
		for (Node node : nodes.values()) {
			int id=node.get_leftId();
			if(id!=-1){
				Node other=nodes.get(new Integer(id));
				node.set_left(other);
			}
			id=node.get_rightId();
			if(id!=-1){
				Node other=nodes.get(new Integer(id));
				node.set_right(other);
			}
		}
				
		return nodes.get(new Integer(1));
		}	
}
