/**
 * ScenarioLauncher for the optimal bidding simulations.
 * 
 * 	Features:
 *	 	- Runs "ntournaments" tournaments. Each tournament has X sessions.
 * 	 		- For each tournament, it dynamically/randomly picks two agents from agentrepository.xml
 * 			- Loads two preferences' profile, 
 *  			- Stores the log in a specific location log/_log
 *  
 *  		For example if we run it for two agents and 3 tournaments at 2013-09-26 18.58.25, the result will be generated
 *  		and stored in 3 log files 2013-09-26 18.58.25__i.xml, with i=1,2,3.
 * 
 * Note:
 *     Before running ScenarioLauncher:
 * 	 	1. first run the SGG to generate the profiles
 * 	 	2. move them to domainrepository.xml (manually or automatically*)
 *	
 *  TODO deactivate the agents involving GUIs...
 *  TODO add the agents' names, rvB and rvA to the prefix of the log file name.
 *  TODO How to fix the number of rounds (X) in a session !?
 *  TODO (*)
 * 
 * @author rafik  
 **************************************************************************************************************/

package genius.core.misc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import genius.core.Global;
import genius.core.protocol.Protocol;
import genius.core.repository.AgentRepItem;
import genius.core.repository.DomainRepItem;
import genius.core.repository.ProfileRepItem;
import genius.core.repository.ProtocolRepItem;
import genius.core.repository.Repository;
import genius.core.repository.RepositoryFactory;
import genius.core.tournament.TournamentConfiguration;

public class ScenarioLauncher {
	public static void main(String[] args) throws Exception {
		try {
			// globals

			int ntournaments = 3; // number of tournaments
			boolean trace = false;
			String path = "file:/Users/rafik/Documents/workspace/NegotiatorGUI voor AAMAS competitie/";
			String p = "negotiator.protocol.alternatingoffers.AlternatingOffersProtocol";
			String outputFile = "log/_" + Global.getOutcomesFileName();

			// Loading the agents from repository

			Repository<AgentRepItem> agentrepository = RepositoryFactory.get_agent_repository();
			ArrayList<AgentRepItem> agents_names = agentrepository.getItems();
			Repository<AgentRepItem> repAgent = RepositoryFactory.get_agent_repository();
			AgentRepItem[] agentsARI = new AgentRepItem[agents_names.size()];

			for (int i = 0; i < agents_names.size(); i++) {
				if ((agentsARI[i] = (AgentRepItem) repAgent.getItemByName(agents_names.get(i).getName())) == null)
					throw new Exception("Unable to createFrom agent " + agents_names.get(i).getName() + "!");

				if (trace)
					System.out.println(" Agent " + i + " :  " + agentsARI[i].getName() + "\n \t "
							+ agentsARI[i].getDescription() + "\n \t " + agentsARI[i].getClassPath() + "\n \t "
							+ agentsARI[i].getParams() + "\n \t " + agentsARI[i].getVersion());
			}

			// Loading the preferences' profiles from repository

			Repository<DomainRepItem> domainrepository = RepositoryFactory.get_domain_repos();
			ArrayList<DomainRepItem> profiles_names = RepositoryFactory.get_domain_repos().getItems();
			DomainRepItem[] DomainsARI = new DomainRepItem[profiles_names.size()];

			for (int i = 0; i < profiles_names.size(); i++) {
				if ((DomainsARI[i] = (DomainRepItem) domainrepository
						.getItemByName(profiles_names.get(i).getName())) == null)
					throw new Exception("Unable to createFrom domain " + profiles_names.get(i).getName() + "!");

				if (trace)
					System.out.println(" Domain " + i + " :  " + DomainsARI[i].getName() + "\n \t "
							+ DomainsARI[i].getClass() + "\n \t " + DomainsARI[i].getFullName() + "\n \t "
							+ DomainsARI[i].getProfiles() + "\n \t " + DomainsARI[i].getURL());
			}

			// Init tournaments

			Protocol ns;
			Thread[] threads = new Thread[ntournaments];

			for (int i = 1; i <= ntournaments; i++) {
				// In the following, as an example, we randomly pick two
				// different agents
				// another way is to try all the combinations...

				String random_A = agentsARI[(new Random()).nextInt(agentsARI.length)].getClassPath(),
						random_B = new String(random_A);

				while (random_B.equals(random_A))
					random_B = agentsARI[(new Random()).nextInt(agentsARI.length)].getClassPath();

				// ...same for the domains: pick two random preferences'
				// profiles.

				int rand = (new Random()).nextInt(DomainsARI.length);
				String random_Domain = DomainsARI[rand].getURL().toString();
				random_Domain = random_Domain.replaceAll("file:", "");

				System.out.println(" random_Domain : " + random_Domain);
				System.out.println(" Profiles : " + DomainsARI[rand].getProfiles());

				String domainFile = path + random_Domain;
				System.out.println(" domainFile: " + domainFile);
				List<String> profiles = Arrays.asList(path + DomainsARI[rand].getProfiles().toArray()[0], // "etc/templates/laptopdomain/laptop_buyer_utility.xml",
						path + DomainsARI[rand].getProfiles().toArray()[1]); // "etc/templates/laptopdomain/laptop_seller_utility.xml");
				System.out.println(" profiles:  profile 1 : " + profiles.get(0) + "\n\tprofile 2 : " + profiles.get(1));

				List<String> agents = Arrays.asList(random_A, random_B);

				System.out.println(" agents:    agent 1 : " + agents.get(0) + "\n\tagent 2 : " + agents.get(1));

				outputFile = outputFile.replaceAll((i == 1) ? ".xml" : "__(\\d+).xml", "__" + i + ".xml");

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
					throw new IllegalArgumentException("The number of profiles does not match the number of agents!");

				ProtocolRepItem protocol = new ProtocolRepItem(p, p, p);
				System.out.println(" protocol name: " + protocol.getDescription());
				DomainRepItem dom = new DomainRepItem(new URL(domainFile));
				ProfileRepItem[] agentProfiles = new ProfileRepItem[profiles.size()];

				for (int j = 0; j < profiles.size(); j++) {
					agentProfiles[j] = new ProfileRepItem(new URL(profiles.get(j)), dom);
					if (agentProfiles[j].getDomain() != agentProfiles[0].getDomain())
						throw new IllegalArgumentException("Profiles for agent 1 and agent " + (j + 1)
								+ " do not have the same domain. Please correct your profiles");
				}

				AgentRepItem[] agentsrep = new AgentRepItem[agents.size()];
				for (int j = 0; j < agents.size(); j++)
					agentsrep[j] = new AgentRepItem(agents.get(j), agents.get(j), agents.get(j));

				ns = Global.createProtocolInstance(protocol, agentsrep, agentProfiles, null);

				TournamentConfiguration.addOption("deadline", 60);
				TournamentConfiguration.addOption("oneSidedBidding", 0);
				TournamentConfiguration.addOption("startingAgent", 0);
				TournamentConfiguration.addOption("accessPartnerPreferences", 0);
				TournamentConfiguration.addOption("appendModeAndDeadline", 0);
				TournamentConfiguration.addOption("disableGUI", 0);
				TournamentConfiguration.addOption("logNegotiationTrace", 0);
				TournamentConfiguration.addOption("protocolMode", 0); // TODO
				TournamentConfiguration.addOption("allowPausingTimeline", 0);
				TournamentConfiguration.addOption("logFinalAccuracy", 0);
				TournamentConfiguration.addOption("logDetailedAnalysis", 0);
				TournamentConfiguration.addOption("logCompetitiveness", 0);

				System.out.println("======== Tournament " + i + "/" + ntournaments + " started ========{");
				// Set the tournament.
				threads[i - 1] = new Thread(ns);
				threads[i - 1].start();
				threads[i - 1].join(); // wait until the tournament finishes.
				System.out.println("Thread " + i + " finished.");

				System.out.println("   ns.getName()          = " + ns.getName());
				System.out.println("   ns.getSessionNumber() = " + ns.getSessionNumber());
				System.out.println("   ns.getTotalSessions() = " + ns.getTotalSessions());
				System.out.println("======== Tournament " + i + "/" + ntournaments + " finished ========}");

			} // i

			System.out.println("\n" + ntournaments + " tournaments finished.");

		} catch (Exception e) {
			e.printStackTrace();
		}

	} // end main

} // end ScenarioLauncher
