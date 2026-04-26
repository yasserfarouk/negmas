/****
 * Singleton <SimulatorConfiguration>
 * Loads the configuration parameters for <Simulator> from "simulatorrepository.xml"
 * @author rafik
 *******************************************************************************************************/
package genius.core.misc;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;

public class SimulatorConfiguration 
{
	private static SimulatorConfiguration sSimConfInstance = null;

	static String[] tags = {"simulator", "rounds", "agents", "tournament"};
	static String file;
	HashMap<String, String> config, tournament;
	ArrayList<String>       agents;
	ArrayList<Integer>     rounds;

	private SimulatorConfiguration(String confile)
	{
		try 
		{
			file = confile;
			DocumentBuilderFactory domFactory = DocumentBuilderFactory.newInstance(); 
			domFactory.setIgnoringComments(true);
			DocumentBuilder builder = domFactory.newDocumentBuilder(); 
			Document doc = builder.parse(new File(file)); 
			doc.getDocumentElement().normalize();

			NodeList nodes = doc.getElementsByTagName(tags[0]);
			
			config = new HashMap<String, String>();

			for (int i = 0 ; i < nodes.item(0).getAttributes().getLength() ; i++)
				config.put(nodes.item(0).getAttributes().item(i).getNodeName(), nodes.item(0).getAttributes().item(i).getNodeValue());

			for (int i = 0 ; i < nodes.item(0).getChildNodes().getLength() ; i++)
			{
				Node node = nodes.item(0).getChildNodes().item(i);
			
				if (node.getNodeType() == Node.ELEMENT_NODE)
				{
						if (node.getNodeName().equals(tags[1]))
						{
							rounds = new ArrayList<Integer>();
							for (int j=0 ; j < node.getChildNodes().getLength() ; j++)
								if (node.getChildNodes().item(j).getNodeType() == Node.ELEMENT_NODE)
									rounds.add(Integer.parseInt(node.getChildNodes().item(j).getTextContent()));
						}
						
						if (node.getNodeName().equals(tags[2]))
						{
							agents = new ArrayList<String>();
							for (int j=0 ; j < node.getChildNodes().getLength() ; j++)
								if (node.getChildNodes().item(j).getNodeType() == Node.ELEMENT_NODE)
									agents.add(node.getChildNodes().item(j).getTextContent());
						}
						if (node.getNodeName().equals(tags[3])) // tournament
						{
							tournament = new HashMap<String, String>();
							for (int j=0 ; j < node.getChildNodes().getLength() ; j++)
								if (node.getChildNodes().item(j).getNodeType() == Node.ELEMENT_NODE)
									tournament.put(node.getChildNodes().item(j).getAttributes().item(0).getNodeValue(),
												  node.getChildNodes().item(j).getTextContent());
						}
				}
			}
		}
		catch (SAXParseException err) 
		{
			System.out.println("Parsing error" + ", line " + err.getLineNumber () + ", uri " + err.getSystemId ());
			System.out.println(" " + err.getMessage ());
		}
		catch (SAXException e) 
		{
			Exception x = e.getException ();
			((x == null) ? e : x).printStackTrace ();
		}
		catch (Throwable t) 
		{
			t.printStackTrace ();
		}
	} 

	public static SimulatorConfiguration getInstance(String s)
	{
		 if (sSimConfInstance == null) 
		 	 sSimConfInstance = new SimulatorConfiguration(s);
	     
		 return sSimConfInstance;
	}
	public ArrayList<?> get(String option)
	{
		if (option.equals(tags[2]))			return agents;
		else if (option.equals(tags[1]))	    return rounds;
		System.out.println("Wrong configuration parameter!");
		return null;
	}

	public HashMap<String, String> getConf()
	{
		return config;
	}
	public HashMap<String, String> getTournamentOptions()
	{
		return tournament;
	}

    public static void main (String argv [])
    {
    		SimulatorConfiguration conf =  SimulatorConfiguration.getInstance("/Users/rafik/Documents/workspace/NegotiatorGUI voor AAMAS competitie/simulatorrepository.xml"); 

    		  for (String key : conf.getTournamentOptions().keySet())
    			  System.out.println( key + " : " + conf.getTournamentOptions().get(key));
    		
    		//  for (String key : conf.getConf().keySet())
    		//      System.out.println( key + " : " + conf.getConf().get(key));
    		  
    		//  for (Object i : conf.get("rounds"))
    		//      System.out.println( "round = " + i );

    		//  for (Object i : conf.get("agents"))
    		//      System.out.println( "agent : " + i );

    	} // main

}
