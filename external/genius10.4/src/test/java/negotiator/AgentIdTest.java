package negotiator;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import genius.core.AgentID;

@RunWith(Parameterized.class)
public class AgentIdTest {
	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] { { "Agent@1", "Agent" }, { "Agent@library", "Agent@library" },
				{ "Agent@1@2", "Agent@1" } });
	}

	private String input;
	private String expected;

	public AgentIdTest(String input, String expected) {
		this.input = input;
		this.expected = expected;
	}

	@Test
	public void test1() {
		assertEquals(expected, new AgentID(input).getName());
	}
}
