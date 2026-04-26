import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import java.io.FileInputStream;
import java.lang.Class;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import py4j.GatewayServer;
import java.io.Serializable;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import negotiator.AgentID;
import negotiator.Bid;
import negotiator.Deadline;
import negotiator.DeadlineType;
import negotiator.Domain;
import negotiator.DomainImpl;
import negotiator.actions.Accept;
import negotiator.actions.Inform;
import negotiator.actions.Action;
import negotiator.actions.EndNegotiation;
import negotiator.actions.Offer;
import negotiator.issue.Issue;
import negotiator.issue.IssueDiscrete;
import negotiator.issue.Value;
import negotiator.issue.ValueDiscrete;
import negotiator.parties.AbstractNegotiationParty;
import negotiator.parties.NegotiationInfo;
import negotiator.persistent.DefaultPersistentDataContainer;
import negotiator.persistent.PersistentDataType;
import negotiator.timeline.ContinuousTimeline;
import negotiator.timeline.DiscreteTimeline;
import negotiator.timeline.TimeLineInfo;
import negotiator.utility.AbstractUtilitySpace;
import negotiator.utility.AdditiveUtilitySpace;
import negotiator.Agent;


import java.util.Map;

class MapUtil {
    public static String mapToString(Map<String, String> map
            , String entry_separator, String internal_separator) {
        StringBuilder stringBuilder = new StringBuilder();

        for (String key : map.keySet()) {
            if (stringBuilder.length() > 0) {
                stringBuilder.append(entry_separator);
            }
            String value = map.get(key);
            //try {
                //stringBuilder.append((key != null ? URLEncoder.encode(key, "UTF-8") : ""));
                stringBuilder.append((key != null ? key : ""));
                stringBuilder.append(internal_separator);
                stringBuilder.append(value != null ? value : "");
                //stringBuilder.append(value != null ? URLEncoder.encode(value, "UTF-8") : "");
            //} catch (UnsupportedEncodingException e) {
            //    throw new RuntimeException("This method requires UTF-8 encoding support", e);
           // }
        }

        return stringBuilder.toString();
    }

    public static Map<String, String> stringToMap(String input
            , String entry_separator, String internal_separator ) {
        Map<String, String> map = new HashMap<String, String>();

        String[] nameValuePairs = input.split(entry_separator);
        for (String nameValuePair : nameValuePairs) {
            String[] nameValue = nameValuePair.split(internal_separator);
            //try {
                //map.put(URLDecoder.decode(nameValue[0], "UTF-8"), nameValue.length > 1 ? URLDecoder.decode(
                //        nameValue[1], "UTF-8") : "");
                map.put(nameValue[0], nameValue.length > 1 ?  nameValue[1] : "");
            //} catch (UnsupportedEncodingException e) {
            //    throw new RuntimeException("This method requires UTF-8 encoding support", e);
            //}
        }

        return map;
    }
}

class NegLoader{
    public String jarName = "genius-8.0.4-jar-with-dependencies.jar";
    private HashMap<String, AbstractNegotiationParty> parties = null;
    private HashMap<String, Agent> agents = null;
    private HashMap<String, Boolean> is_party = null;
    private HashMap<String, NegotiationInfo> infos = null;
    private HashMap<String, AgentID> ids = null;
    private HashMap<String, Domain> domains = null;
    private HashMap<String, AdditiveUtilitySpace> util_spaces = null;
    private HashMap<String, Boolean> first_actions = null;

    private HashMap<String, ArrayList<Issue> > issues_all;
    private HashMap<String, HashMap<String, HashMap<String, Value> > > string2values = null;
    private HashMap<String, HashMap<String, Issue > > string2issues = null;
    private HashMap<String , TimeLineInfo> timelines = null;

    private int n_agents = 0;

    private String INTERNAL_SEP = "<<s=s>>";
    private String ENTRY_SEP = "<<y,y>>";
    private String FIELD_SEP = "<<sy>>";

    public class Serialize implements Serializable{
    }

    public NegLoader(){
        parties = new HashMap<String, AbstractNegotiationParty>();
        agents = new HashMap<String, Agent>();
        is_party = new HashMap<String, Boolean>();
        infos = new HashMap<String, NegotiationInfo>();
        ids = new HashMap<String, AgentID>();
        domains  = new HashMap<String, Domain>();
        util_spaces  = new HashMap<String, AdditiveUtilitySpace>();
        first_actions = new HashMap<String, Boolean>();
        string2values = new HashMap<String, HashMap<String, HashMap<String, Value> > >();
        issues_all = new HashMap<String, ArrayList<Issue> >();
        timelines = new HashMap<String , TimeLineInfo>();
        string2issues = new HashMap<String, HashMap<String, Issue > >();
    }

    public String test(String class_name){
        ArrayList classes = new ArrayList();

        System.out.println("Jar " + jarName );
        try {
            JarInputStream jarFile = new JarInputStream(new FileInputStream(
                    jarName));
            JarEntry jarEntry;

            while (true) {
                jarEntry = jarFile.getNextJarEntry();
                if (jarEntry == null) {
                    break;
                }
                if (jarEntry.getName().endsWith(".class")) {
                    //System.out.println("Found "
                    //        + jarEntry.getName().replaceAll("/", "\\."));
                    classes.add(jarEntry.getName().replaceAll("/", "\\."));
                }
            }
            Class<?> clazz = Class.forName(class_name);
            System.out.println(clazz.toString());
            //Constructor<?> constructor = clazz.getConstructor(String.class, Integer.class);
            AbstractNegotiationParty instance = (AbstractNegotiationParty) clazz.newInstance();
            return instance.getDescription();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }


    private NegotiationInfo createNegotiationInfo(String domain_file_name,
                                          String utility_file_name, boolean real_time
            , int max_time, long seed, String agent_uuid) {
        try {
            DomainImpl domain = new DomainImpl(domain_file_name);
            AdditiveUtilitySpace utilSpace = new AdditiveUtilitySpace(domain, utility_file_name);
            TimeLineInfo timeline;
            DeadlineType tp;
            if(real_time) {
                tp = DeadlineType.TIME;
                timeline = new ContinuousTimeline(max_time);
            }else {
                tp = DeadlineType.ROUND;
                timeline = new DiscreteTimeline(max_time);
            }
            Deadline deadline = new Deadline(max_time, tp);
            long randomSeed = seed;
            AgentID agentID = new AgentID(agent_uuid);
            DefaultPersistentDataContainer storage = new DefaultPersistentDataContainer(new Serialize(), PersistentDataType.DISABLED);

            NegotiationInfo info = new NegotiationInfo(utilSpace, deadline, timeline, randomSeed, agentID, storage);
            infos.put(agent_uuid, info);
            ids.put(agent_uuid, agentID);
            util_spaces.put(agent_uuid, utilSpace);
            domains.put(agent_uuid, domain);
            timelines.put(agent_uuid, timeline);
            first_actions.put(agent_uuid, true);
            ArrayList<Issue> issues = (ArrayList<Issue>) utilSpace.getDomain().getIssues();
            issues_all.put(agent_uuid, issues);
            string2values.put(agent_uuid, this.init_str_val_conversion(agent_uuid, issues));
            HashMap<String, Issue> striss = new HashMap<String, Issue>();
            for(Issue issue:issues){
                striss.put(issue.toString(), issue);
            }
            string2issues.put(agent_uuid, striss);
            return info;
        } catch (Exception e) {
            // TODO: handle exception
            System.out.println(e);
        }
        return null;
    }

    public void on_negotiation_start(
            String agent_uuid,
            int n_agents,
            long n_steps,
            long time_limit,
            boolean real_time,
            String domain_file_name,
            String utility_file_name
    ){
        this.n_agents = n_agents;
        if (is_party.get(agent_uuid)) {
            AbstractNegotiationParty agent = this.parties.get(agent_uuid);
            NegotiationInfo info = createNegotiationInfo(domain_file_name,
                    utility_file_name, real_time
                    , real_time ? (int) time_limit : (int) n_steps
                    , 0, agent_uuid);
            if (info == null)
                return;
            parties.get(agent_uuid).init(info);
        }else{
            Agent agent = this.agents.get(agent_uuid);
            agent.init();
        }
        System.out.format("Agent %s: time limit %d, step limit %d\n", getName(agent_uuid)
            , time_limit, n_steps);
    }

    public String create_agent(String class_name){
        try {
            //JarInputStream jarFile = new JarInputStream(new FileInputStream(
            //        jarName));
            Class<?> clazz = Class.forName(class_name);
            String uuid = class_name + UUID.randomUUID().toString();
            //Constructor<?> constructor = clazz.getConstructor(String.class, Integer.class);
            if (AbstractNegotiationParty.class.isAssignableFrom(clazz)) {
                AbstractNegotiationParty agent = (AbstractNegotiationParty) clazz.newInstance();
                this.is_party.put(uuid, true);
                this.parties.put(uuid, agent);

            }else{
                Agent agent = (Agent) clazz.newInstance();
                this.is_party.put(uuid, false);
                this.agents.put(uuid, agent);
            }
            return uuid;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return "";
    }

	public String destroy_agent(String agent_uuid){
		this.infos.remove(agent_uuid);
		this.ids.remove(agent_uuid);
		this.util_spaces.remove(agent_uuid);
		this.domains.remove(agent_uuid);
		this.timelines.remove(agent_uuid);
		this.first_actions.remove(agent_uuid);
		this.string2values.remove(agent_uuid);
		this.string2issues.remove(agent_uuid);
		this.issues_all.remove(agent_uuid);
		if(is_party.get(agent_uuid)){
			this.parties.remove(agent_uuid);
		}else{
			this.agents.remove(agent_uuid);
		}
		return "";
	}

    public String actionToString(Action action){
        String id = action.getAgent().getName();
        Bid bid = null;
        if(action instanceof Offer){
            bid = ((Offer) action).getBid();
            List<Issue> issues = bid.getIssues();
            HashMap<Integer, Value> vals = bid.getValues();
            HashMap<String, String> vals_str = new HashMap<String, String>();

            for (Integer key : vals.keySet()) {
                vals_str.put(issues.get(key-1).toString(), vals.get(key).toString());
            }

            String bidString = MapUtil.mapToString(vals_str, ENTRY_SEP
            , INTERNAL_SEP);
            return id + FIELD_SEP + "Offer" + FIELD_SEP + bidString;
        }
        else if(action instanceof EndNegotiation)
            return id + FIELD_SEP + "EndNegotiation" + FIELD_SEP;
        else if(action instanceof Accept)
            return id + FIELD_SEP + "Accept" + FIELD_SEP;
        return id + FIELD_SEP + "Failure" + FIELD_SEP;
    }

    public String choose_action(String agent_uuid) {
        if (is_party.get(agent_uuid))
            return choose_action_party(agent_uuid);
        return choose_action_agent(agent_uuid);
    }
    public String choose_action_agent(String agent_uuid) {
        Agent agent = agents.get(agent_uuid);
        boolean isFirstTurn = first_actions.get(agent_uuid);
        TimeLineInfo timeline = timelines.get(agent_uuid);
        List<Class<? extends Action>> validActions = new ArrayList<Class<? extends Action>>();
        if(!isFirstTurn)
            validActions.add(Accept.class);
        validActions.add(EndNegotiation.class);
        validActions.add(Offer.class);
        negotiator.actions.Action action = agent.chooseAction(validActions);
        if(timeline instanceof DiscreteTimeline)
            ((DiscreteTimeline) timeline).increment();
        return actionToString(action);

    }
    public String choose_action_party(String agent_uuid) {
        AbstractNegotiationParty agent = parties.get(agent_uuid);
        boolean isFirstTurn = first_actions.get(agent_uuid);
        TimeLineInfo timeline = timelines.get(agent_uuid);
        List<Class<? extends Action>> validActions = new ArrayList<Class<? extends Action>>();
        if(!isFirstTurn)
            validActions.add(Accept.class);
        validActions.add(EndNegotiation.class);
        validActions.add(Offer.class);
        negotiator.actions.Action action = agent.chooseAction(validActions);
        if(timeline instanceof DiscreteTimeline)
            ((DiscreteTimeline) timeline).increment();
        return actionToString(action);
    }
    public Boolean receive_message(String agent_uuid, String from_id
            , String typeOfAction, String bid_str){
        if (is_party.get(agent_uuid))
            return receive_message_party(agent_uuid, from_id, typeOfAction, bid_str);
        return receive_mesasge_agent(agent_uuid, from_id, typeOfAction, bid_str);
    }

    public Boolean receive_mesasge_agent(String agent_uuid, String from_id
            , String typeOfAction, String bid_str) {
        Agent agent = agents.get(agent_uuid);
        boolean isFirstTurn = first_actions.get(agent_uuid);
        if(isFirstTurn)
            first_actions.put(agent_uuid, false);
        Bid bid = str2bid(agent_uuid, bid_str);
        AgentID agentID = new AgentID(from_id);
        Action act = null;
        if (typeOfAction.contains("Offer")){
            act = new Offer(agentID, bid);
        }
        else if(typeOfAction.contains("Accept")){
            act = new Accept(agentID, bid);
        }
        else if(typeOfAction.contains("EndNegotiation"))
            act = new EndNegotiation(agentID);
        agent.ReceiveMessage(act);
        return true;
    }


    public Boolean receive_message_party(String agent_uuid, String from_id
            , String typeOfAction, String bid_str){
        AbstractNegotiationParty agent = parties.get(agent_uuid);
        boolean isFirstTurn = first_actions.get(agent_uuid);
        if(isFirstTurn)
            first_actions.put(agent_uuid, false);
        Bid bid = str2bid(agent_uuid, bid_str);
        AgentID agentID = new AgentID(from_id);
        Action act = null;
        if (typeOfAction.contains("Offer")){
            act = new Offer(agentID, bid);
        }
        else if(typeOfAction.contains("Accept")){
            act = new Accept(agentID, bid);
        }
        else if(typeOfAction.contains("EndNegotiation"))
            act = new EndNegotiation(agentID);
        agent.receiveMessage(agentID, act);
        return true;
    }

    public HashMap<String, HashMap<String, Value>> init_str_val_conversion(String agent_uuid, ArrayList<Issue> issues){
        HashMap<String, HashMap<String, Value>> string2value = new HashMap<String, HashMap<String, Value>> ();
        for(Issue issue:issues){
            String issue_name = issue.toString();
            string2value.put(issue_name, new HashMap<String, Value>());
            List<ValueDiscrete> values = ((IssueDiscrete)issue).getValues();
            for(Value value:values){
                string2value.get(issue_name).put(value.toString(), value);
            }
        }
        return string2value;
    }


    public Bid str2bid(String agent_uuid, String bid_str){
        AbstractUtilitySpace utilSpace = util_spaces.get(agent_uuid);
        ArrayList<Issue> issues = issues_all.get(agent_uuid);
        //Bid bid = new Bid(utilSpace.getDomain());//.getRandomBid(new Random());
        if(bid_str.equals(""))
            return null;
        String[] bid_strs = bid_str.split(ENTRY_SEP);
        HashMap<Integer, Value> vals = new HashMap<Integer, Value>();
        for(String str:bid_strs) {
            String[] vs = str.split(INTERNAL_SEP);
            String issue_name = vs[0];
            String val = vs.length > 1? vs[1]: "";
            vals.put(string2issues.get(agent_uuid).get(issue_name).getNumber(),
                    string2values.get(agent_uuid).get(issue_name).get(val));
        }
        return new Bid(utilSpace.getDomain(), vals);
    }


    public void informMessage(String agent_uuid, int agent_num){
        Inform inform = new Inform(ids.get(agent_uuid), "NumberOfAgents", agent_num);
        parties.get(agent_uuid).receiveMessage(ids.get(agent_uuid), inform);
    }
    public void informMessage(String agent_uuid){
        Inform inform = new Inform(ids.get(agent_uuid), "NumberOfAgents", this.n_agents);
        parties.get(agent_uuid).receiveMessage(ids.get(agent_uuid), inform);
    }
    public String getName(String agent_uuid){
        if (this.is_party.get(agent_uuid))
            return parties.get(agent_uuid).getDescription();
        else
            return agents.get(agent_uuid).getDescription();
    }




    public static void main(String[] args) {
        int port = 25333;
        boolean dieOnBrokenPipe=false;
        System.out.format("Received options: ");
        for (int i = 0; i < args.length; i++) {
            String opt = args[i];
            System.out.format("%s ", opt);
            if (opt.equals("--die-on-exit")) {
                dieOnBrokenPipe = true;
            } else {
                port = Integer.parseInt(opt);
            }
        }
        System.out.format("\n");
        NegLoader app = new NegLoader();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(app, port);
        server.start();
        int listening_port = server.getListeningPort();
        System.out.format("Gateway to python started at port %d listening to port %d\n", port, listening_port);

        if (dieOnBrokenPipe) {
            /* Exit on EOF or broken pipe.  This ensures that the server dies
             * if its parent program dies. */
            try {
                BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in, Charset.forName("UTF-8")));
                stdin.readLine();
                System.exit(0);
            } catch (java.io.IOException e) {
                System.exit(1);
            }
        }

    }
}
