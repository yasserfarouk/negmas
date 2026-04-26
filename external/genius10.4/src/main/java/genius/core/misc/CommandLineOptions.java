package genius.core.misc;

import java.util.List;
import joptsimple.OptionParser;
import joptsimple.OptionSet;

/**
 * This class is used to interpret commandline parameters specified when starting Genius.
 */
public class CommandLineOptions
{
	/** Option "s", automatically open new tournament tab on start up. */
	public boolean newTournament = false;
	/** Option "t", automatically start tournament on start up. */
	public boolean startTournament = false;
	/** Option "q", automatically quit after the tournament finished. */
	public boolean quitWhenTournamentDone = false;
	/** Option "a", specify a list of agents for the commandline runner. */
	public List<String> agents;
	/** Option "p", specify a list of profiles for the commandline runner. */
	public List<String> profiles;
	/** Option "r", specify a protocol for the commandline runner. */
	public String protocol = "negotiator.protocol.alternatingoffers.AlternatingOffersProtocol";
	/** Option "d", specify a domain. */
	public String domain;
	/** Option "f", specify the output file for the commandline runner. */
	public String outputFile;

	/**
	 * Method used to parse the commandline options.
	 * @param args arguments given to the commandline.
	 */
	@SuppressWarnings("unchecked")
	public void parse(String [] args)
	{
		OptionParser parser = new OptionParser( "stq:q::a:p:d:f:r:" );
        OptionSet options = parser.parse(args);

        if (options.has("s"))
        	startTournament = true;
        if (options.has("t"))
        	newTournament = true;
        if (options.has("q"))
        	quitWhenTournamentDone = true;
        if (options.has("a"))
        	agents = (List<String>) options.valuesOf("a");
        if (options.has("r"))
        	protocol = (String) options.valueOf("r");
        if (options.has("p"))
        	profiles = (List<String>) options.valuesOf("p");
        if (options.has("d"))
        	domain = (String) options.valueOf("d");
        if (options.has("f"))
        	outputFile = (String) options.valueOf("f");
	}
}