/*************************************************************************************************************
 * Simulator for the optimal bidding experiments.
 * 
 *	 - Generates all the combinations agents/profiles and runs trials
 * 	 - For each trial it stores the log in log/_log
 *     E.g., for two agents and 3 trials at 2013-09-26 18.58.25, the result will be generated 
 *     and stored in 3 log files: 2013-09-26 18.58.25__i.xml, with i=1,2,3.
 * 
 * 	 - Before running Simulator we first need to run SGG to generate the profiles
 *	
 *  TODO Fix the number of rounds (X) in a session !
 * 
 * @author rafik hadfi 
 **************************************************************************************************************/

package genius.core.misc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import genius.core.Global;
import genius.core.protocol.Protocol;
import genius.core.repository.AgentRepItem;
import genius.core.repository.DomainRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.ProtocolRepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.core.tournament.TournamentConfiguration;

public class Simulator {
	public static void main(String[] args) throws Exception {
		try {
			// Loading simulator's configuration

			SimulatorConfiguration conf = SimulatorConfiguration.getInstance(Global.SIMULATOR_REPOSITORY);

			String path = conf.getConf().get("root"), ProtoName = conf.getConf().get("protocol"),
					outputFile = conf.getConf().get("log") + Global.getOutcomesFileName().replaceAll("log/", "");
			boolean trace = conf.getConf().get("trace").equals("false") ? false : true,
					all = conf.getConf().get("all").equals("false") ? false : true; // if
																					// ==
																					// true,
																					// totaltrials==number
																					// of
																					// all
																					// possible
																					// combinations
			int totaltrials = Integer.parseInt(conf.getConf().get("trials"));

			// Trials' variables

			boolean first = true;
			int trial = 1; // number of trials ~ tournaments
			Protocol ns = null;
			// ~~~Set<Set<RepItem>> AgentsCombinations = null;
			Set<Set<String>> AgentsCombinations = null;
			// ~~~ Iterator<Set<RepItem>> combination = null;
			Iterator<Set<String>> combination = null;

			List<String> profiles = null;

			// Loading the agents (classes) from repository

			Repository<AgentRepItem> repAgent = RepositoryFactory.get_agent_repository();
			ArrayList<String> agentsList = (ArrayList<String>) conf.get("agents");
			AgentRepItem[] agentsARI = new AgentRepItem[agentsList.size()];

			for (int i = 0; i < agentsList.size(); i++) {
				if (agentsList.contains((String) repAgent.getItemByName(agentsList.get(i)).getName())) {
					agentsARI[i] = (AgentRepItem) repAgent.getItemByName(agentsList.get(i));

					if (agentsARI[i] == null)
						throw new Exception("Unable to createFrom agent " + agentsList.get(i) + "!");
					if (trace)
						System.out.println(" Agent #" + (i + 1) + "/" + agentsList.size() + "\n \t agent name     :  "
								+ agentsARI[i].getName() + "\n \t descr class    :  " + agentsARI[i].getDescription()
								+ "\n \t class path     :  " + agentsARI[i].getClassPath() + "\n \t param profiles :  "
								+ agentsARI[i].getParams() + "\n \t version        :  " + agentsARI[i].getVersion());
				}
			}

			// Loading the preferences' profiles from repository

			Repository<DomainRepItem> domainrepository = RepositoryFactory.get_domain_repos();
			ArrayList<DomainRepItem> profiles_names = domainrepository.getItems();
			ArrayList<DomainRepItem> Domains__ = new ArrayList<DomainRepItem>();

			for (int i = 0; i < profiles_names.size(); i++) {
				DomainRepItem d_ = (DomainRepItem) domainrepository.getItemByName(profiles_names.get(i).getName());
				if (d_ == null)
					throw new Exception("Unable to createFrom domain " + profiles_names.get(i).getName() + "!");

				if (d_.toString().substring(0, 3).equals(conf.getConf().get("domain")))
					Domains__.add(((DomainRepItem) domainrepository.getItems().get(i)));
			}

			DomainRepItem[] DomainsARI = new DomainRepItem[Domains__.size()];
			for (int j = 0; j < Domains__.size(); j++) {
				DomainsARI[j] = Domains__.get(j);
				if (true)
					System.out.println("\n Domain #" + (j + 1) + "/" + Domains__.size() + "  Domain name     :  "
							+ DomainsARI[j].getName() + "\n \t Domain class    :  " + DomainsARI[j].getClass()
							+ "\n \t Domain fullname :  " + DomainsARI[j].getFullName() + "\n \t Domain profiles :  "
							+ DomainsARI[j].getProfiles() + "\n \t Domain URL      :  " + DomainsARI[j].getURL());
			}

			// All the combinations

			// ~~~ Set<RepItem> NamesSet = new HashSet<RepItem>();
			// ~~~ Iterator<RepItem> iter = agents_names.iterator();
			Iterator<String> iter = agentsList.iterator();
			Set<String> NamesSet = new HashSet<String>();

			while (iter.hasNext())
				NamesSet.add(iter.next());

			AgentsCombinations = SetTools.cartesianProduct(NamesSet, NamesSet);

			System.out.println("\n Total [Agents] combinations  : " + AgentsCombinations.size()
					+ "\n Total [Preferences] profiles : " + DomainsARI.length);

			// }}

			System.out.println(
					"=========== runs ======================================================================================================================================================");

			// ####### trials ################################################

			if (all) // all combinations
				totaltrials = AgentsCombinations.size() * DomainsARI.length;

			// T> combination = AgentsCombinations.iterator();
			// T> while (combination.hasNext())
			// T> System.out.println("\t > "+ combination.next());
			// T> System.out.println("\t totaltrials = "+ totaltrials );

			for (DomainRepItem domain : DomainsARI) {
				combination = AgentsCombinations.iterator();

				while (combination.hasNext()) {
					System.out.println("======== Trial " + trial + "/" + totaltrials + " started ==========={");

					String domainFile = path + domain.getURL().toString().replaceAll("file:", "");
					System.out.println(" domainFile: " + domainFile);

					profiles = Arrays.asList(path + domain.getProfiles().toArray()[0],
							path + domain.getProfiles().toArray()[1]);
					System.out.println(
							" profiles:  profile 1 : " + profiles.get(0) + "\n\tprofile 2 : " + profiles.get(1));

					Object[] tc = combination.next().toArray();

					// ~~~ String AClassPath = new String(tc[0].toString());
					// ~~~ String BClassPath = new String(tc[(tc.length==1) ? 0
					// : 1].toString()); // play against self

					String AClassPath = null, BClassPath = null;

					for (int d = 0; d < agentsARI.length; d++) {
						if (agentsARI[d].getName().equals(tc[0].toString()))
							AClassPath = new String(agentsARI[d].getClassPath());
						if (agentsARI[d].getName().equals(tc[(tc.length == 1) ? 0 : 1].toString())) // play
																									// against
																									// self
							BClassPath = new String(agentsARI[d].getClassPath());
					}

					List<String> agents = Arrays.asList(AClassPath, BClassPath);
					System.out.println(" agents:    agent 1 : " + agents.get(0) + "\n\tagent 2 : " + agents.get(1));

					if (first) {
						outputFile = outputFile.replaceAll(".xml", "__" + trial + ".xml");
						first = false;
					} else
						outputFile = outputFile.replaceAll("__(\\d+).xml", "__" + trial + ".xml");

					File outcomesFile = new File(outputFile);
					BufferedWriter out = new BufferedWriter(new FileWriter(outcomesFile, true));
					if (!outcomesFile.exists()) {
						System.out.println("Creating log file " + outputFile);
						out.write("<a>\n");
					}
					out.close();
					System.out.println(" logfile: " + outputFile);
					Global.logPreset = outputFile;
					if (profiles.size() != agents.size())
						throw new IllegalArgumentException(
								"The number of profiles does not match the number of agents!");

					ProtocolRepItem protocol = new ProtocolRepItem(ProtoName, ProtoName, ProtoName);
					DomainRepItem dom = new DomainRepItem(new URL(domainFile));
					ProfileRepItem[] agentProfiles = new ProfileRepItem[profiles.size()];

					System.out.println(" protocol name: " + protocol.getDescription());

					for (int j = 0; j < profiles.size(); j++) {
						agentProfiles[j] = new ProfileRepItem(new URL(profiles.get(j)), dom);
						if (agentProfiles[j].getDomain() != agentProfiles[0].getDomain())
							throw new IllegalArgumentException("Profiles for agent 1 and agent " + (j + 1)
									+ " do not have the same domain. Please correct your profiles");
					}

					AgentRepItem[] agentsrep = new AgentRepItem[agents.size()];
					for (int j = 0; j < agents.size(); j++)
						agentsrep[j] = new AgentRepItem(agents.get(j), agents.get(j), agents.get(j));

					System.out.print("\n Loading options...\n");
					for (String option : conf.getTournamentOptions().keySet())
						TournamentConfiguration.addOption(option,
								Integer.parseInt(conf.getTournamentOptions().get(option)));

					// negotiation instance

					ns = Global.createProtocolInstance(protocol, agentsrep, agentProfiles, null);
					System.out.print("Negotiation session built: " + ns + "\n");
					ns.startSession();

					System.out.print("...\n");
					Thread.sleep(500);

					System.out.println(" \t   ns.getName()          = " + ns.getName());
					System.out.println(" \t   ns.getSessionNumber() = " + ns.getSessionNumber());
					System.out.println(" \t   ns.getTotalSessions() = " + ns.getTotalSessions());

					System.out.println("======== Trial " + trial + "/" + totaltrials + " finished ========} \n");

					if (trial == totaltrials && all == false) {
						System.out.println("\n" + trial + "/" + totaltrials + " trials finished.");
						System.exit(0);
					}

					trial++;

				} // combination
			} // domain

			System.out.println("\n" + (trial - 1) + " trials finished from " + totaltrials + " combinations");

		} catch (Exception e) {
			e.printStackTrace();
		}

	} // end main

} // end Simulator
