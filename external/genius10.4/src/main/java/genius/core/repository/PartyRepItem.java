package genius.core.repository;

import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;

import agents.rlboa.RLBOA;
import genius.core.Agent;
import genius.core.Global;
import genius.core.exceptions.InstantiateException;
import genius.core.parties.Mediator;
import genius.core.parties.NegotiationParty;

/**
 * This repository item contains all info about a {@link NegotiationParty} that
 * can be loaded. immutable.
 */

@SuppressWarnings("serial")
@XmlRootElement(name = "party")
public class PartyRepItem extends ParticipantRepItem {

	private final static String FAILED = "FAILED TO LOAD";
	/**
	 * This can be two things:
	 * <ul>
	 * <li>a class path, eg "agents.anac.y2010.AgentFSEGA.AgentFSEGA". In this
	 * case, the agent must be on the class path to load.
	 * <li>a full path, eg
	 * "/Volumes/documents/NegoWorkspace3/NegotiatorGUI/src/agents/anac/y2010/AgentFSEGA/AgentFSEGA.java"
	 * . In this case, we can figure out the class path ourselves and load it.
	 * </ul>
	 */
	@XmlAttribute
	protected String classPath = "";
	
	@XmlAttribute 
	protected String strategyParameters = "";

	/**
	 * Name, also a short version of the class path.
	 */
	private String partyName = FAILED;

	/**
	 * description of this agent. Cached, Not saved to XML.
	 */
	private String description = FAILED;

	/**
	 * True if this party is a mediator.
	 */
	private Boolean isMediator = false;

	/**
	 * needed to support XML de-serialization.
	 * 
	 * @throws InstantiateException
	 */
	@SuppressWarnings("unused")
	private PartyRepItem() throws InstantiateException {
	}

	/**
	 * 
	 * @param path
	 *            full.path.to.class or file name.
	 */
	public PartyRepItem(String path) {
		if (path == null) {
			throw new NullPointerException(path = null);
		}
		classPath = path;
	}

	/**
	 * Init our fields to cache the party information. party must have been set
	 * before getting here.
	 * 
	 * @param party
	 * @throws InstantiateException
	 */
	@Override
	protected NegotiationParty init() throws InstantiateException {
		NegotiationParty party1 = super.init();
		partyName = getRepName(party1);
		description = party1.getDescription();
		isMediator = party1 instanceof Mediator;
		return party1;
	}

	private String getRepName(NegotiationParty party) 
	{
		String name = party.getClass().getSimpleName();
		// Agents can implement getName() for a better party name
		if (party instanceof Agent)
		{
			String agentName = ((Agent) party).getName();
			if (agentName != null && !"".equals(agentName))
				name = agentName;
		}
		return name;
	}

	/**
	 * @return classpath, either as full package class path (with dots) or as an
	 *         absolute path to a .class file.
	 */
	public String getClassPath() {
		return classPath;
	}

	public String getName() {
		try {
			initSilent();
		} catch (InstantiateException e) {
			return FAILED + " " + classPath;
		}
		return partyName;
	}

	public String getDescription() {
		try {
			initSilent();
		} catch (InstantiateException e) {
			return e.toString();
		}
		return description;
	}

	public String toString() {
		try {
			initSilent();
		} catch (InstantiateException e) {
			return e.toString();
		}
		return "PartyRepositoryItem[" + partyName + "," + classPath + "," + description + ", is mediator="
				+ isMediator().toString() + "]";
	}

	public Boolean isMediator() {
		try {
			initSilent();
		} catch (InstantiateException e) {
			return false; // guess, we can't do much here.
		}
		return isMediator;
	}

	public NegotiationParty load() throws InstantiateException {
		
		NegotiationParty party = (NegotiationParty) Global.loadObject(classPath);
		
		// [09/08/18 jasperon] Added this and the property strategyParameters because we couldn't find
		// any place where this method is called. the check for RLBOA is purposely
		// narrow because of unkown effects on other agents.
		if (party instanceof RLBOA) {
			party = (NegotiationParty) Global.loadAgent(classPath, strategyParameters);
		}
		
		return party;
	}

	/**
	 * @return true if partyName and classPath equal. Note that partyName alone
	 *         is sufficient to be equal as keys are unique.
	 */
	@Override
	public boolean equals(Object o) {
		if (!(o instanceof PartyRepItem))
			return false;
		return classPath.equals(((PartyRepItem) o).classPath);
	}

	@Override
	public String getUniqueName() {
		return Global.shortNameOfClass(classPath);
	}

	@Override
	public String getClassDescriptor() {
		return getClassPath();
	}
}