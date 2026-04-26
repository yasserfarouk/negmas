package negotiator.xml;

import static org.junit.Assert.assertEquals;

import java.io.ByteArrayOutputStream;
import java.util.LinkedHashMap;
import java.util.Map;

import javax.xml.stream.XMLStreamException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import genius.core.xml.Key;
import genius.core.xml.XmlWriteStream;

/**
 * Test (partial) XML stream writing
 */
public class XmlWriteStreamTest {
	private XmlWriteStream stream;
	private ByteArrayOutputStream output;

	@Before
	public void before() throws XMLStreamException {
		output = new ByteArrayOutputStream();
		stream = new XmlWriteStream(output, "test");
	}

	@After
	public void after() {
		System.out.println(output);
	}

	@Test
	public void TestBasicWrite() throws XMLStreamException {
		Map<Object, Object> result = basicResult();
		stream.write("Outcome", result);
		stream.close();

		assertEquals(
				"<?xml version=\"1.0\" ?><test>\n<Outcome currentTime=\"2016-11-15 14:17:17\" timeOfAgreement=\"0.6666666666666666\" lastAction=\"Accept\" v=\"1\" v=\"2\">\n</Outcome></test>",
				output.toString());
		System.out.println(output);
	}

	@Test
	public void TestFullWrite() throws XMLStreamException {
		Map<Object, Object> result = basicResult();

		result.put(new Key("resultOfAgent"), agentResult("A"));
		result.put(new Key("resultOfAgent"), agentResult("B"));

		stream.write("Outcome", result);
		stream.close();

		assertEquals(
				"<?xml version=\"1.0\" ?><test>\n<Outcome currentTime=\"2016-11-15 14:17:17\" timeOfAgreement=\"0.6666666666666666\" lastAction=\"Accept\" v=\"1\" v=\"2\">\n<resultOfAgent agent=\"A\" agentName=\"QAgent\">\n</resultOfAgent>\n<resultOfAgent agent=\"B\" agentName=\"QAgent\">\n</resultOfAgent>\n</Outcome></test>",
				output.toString());
	}

	private Map<Object, Object> basicResult() {
		// use linkedhashmap because the order is important for our exact string
		// assertEquals...
		Map<Object, Object> outcome = new LinkedHashMap<>();
		outcome.put("currentTime", "2016-11-15 14:17:17");
		outcome.put("timeOfAgreement", "0.6666666666666666");
		outcome.put("lastAction", "Accept");
		// check repeated values.
		outcome.put(new Key("v"), 1);
		outcome.put(new Key("v"), 2);
		return outcome;
	}

	private Map<Object, Object> agentResult(String name) {
		Map<Object, Object> outcome = new LinkedHashMap<>();
		outcome.put(new Key("agent"), name);
		outcome.put(new Key("agentName"), "QAgent");
		return outcome;
	}

}
