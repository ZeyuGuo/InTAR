<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="self_attential_spatial" top="selfAttention">
    <files>
        <file name="/home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/self-attn-spatial-testbench.cpp" sc="0" tb="1" cflags=" -I../../include -Wno-unknown-pragmas" csimflags=" -Wno-unknown-pragmas" blackbox="false"/>
        <file name="/home/yingqi/repo/LLM-InTRRA/benchmark/self-attn/self-attn-spatial-kernel.cpp" sc="0" tb="false" cflags="-I./include" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="solution1" status=""/>
    </solutions>
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
</AutoPilot:project>

