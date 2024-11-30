import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
} from "react";
import MotionSegment from "./MotionSegment";
import { ReviewSegment, MotionData } from "@/types/review";

interface VirtualizedMotionSegmentsProps {
  timelineRef: React.RefObject<HTMLDivElement>;
  segments: number[];
  events: ReviewSegment[];
  motion_events: MotionData[];
  segmentDuration: number;
  timestampSpread: number;
  showMinimap: boolean;
  minimapStartTime?: number;
  minimapEndTime?: number;
  contentRef: React.RefObject<HTMLDivElement>;
  setHandlebarTime?: React.Dispatch<React.SetStateAction<number>>;
  dense: boolean;
  motionOnly: boolean;
  getMotionSegmentValue: (timestamp: number) => number;
}

export interface VirtualizedMotionSegmentsRef {
  scrollToSegment: (segmentTime: number, ifNeeded?: boolean) => void;
}

const SEGMENT_HEIGHT = 8;
const OVERSCAN_COUNT = 20;

export const VirtualizedMotionSegments = forwardRef<
  VirtualizedMotionSegmentsRef,
  VirtualizedMotionSegmentsProps
>(
  (
    {
      timelineRef,
      segments,
      events,
      segmentDuration,
      timestampSpread,
      showMinimap,
      minimapStartTime,
      minimapEndTime,
      setHandlebarTime,
      dense,
      motionOnly,
      getMotionSegmentValue,
    },
    ref,
  ) => {
    const [visibleRange, setVisibleRange] = useState({ start: 0, end: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    const updateVisibleRange = useCallback(() => {
      if (timelineRef.current) {
        const { scrollTop, clientHeight } = timelineRef.current;
        const start = Math.max(
          0,
          Math.floor(scrollTop / SEGMENT_HEIGHT) - OVERSCAN_COUNT,
        );
        const end = Math.min(
          segments.length,
          Math.ceil((scrollTop + clientHeight) / SEGMENT_HEIGHT) +
            OVERSCAN_COUNT,
        );
        setVisibleRange({ start, end });
      }
    }, [segments.length, timelineRef]);

    useEffect(() => {
      const container = timelineRef.current;
      if (container) {
        const handleScroll = () => {
          window.requestAnimationFrame(updateVisibleRange);
        };

        container.addEventListener("scroll", handleScroll, { passive: true });
        window.addEventListener("resize", updateVisibleRange);

        updateVisibleRange();

        return () => {
          container.removeEventListener("scroll", handleScroll);
          window.removeEventListener("resize", updateVisibleRange);
        };
      }
    }, [updateVisibleRange, timelineRef]);

    const scrollToSegment = useCallback(
      (segmentTime: number, ifNeeded: boolean = true) => {
        const segmentIndex = segments.findIndex((time) => time === segmentTime);
        if (
          segmentIndex !== -1 &&
          containerRef.current &&
          timelineRef.current
        ) {
          const timelineHeight = timelineRef.current.clientHeight;
          const targetScrollTop = segmentIndex * SEGMENT_HEIGHT;
          const centeredScrollTop =
            targetScrollTop - timelineHeight / 2 + SEGMENT_HEIGHT / 2;

          const isVisible =
            segmentIndex > visibleRange.start + OVERSCAN_COUNT &&
            segmentIndex < visibleRange.end - OVERSCAN_COUNT;

          if (!ifNeeded || !isVisible) {
            timelineRef.current.scrollTo({
              top: Math.max(0, centeredScrollTop),
              behavior: "smooth",
            });
          }
          updateVisibleRange();
        }
      },
      [segments, timelineRef, visibleRange, updateVisibleRange],
    );

    useImperativeHandle(ref, () => ({
      scrollToSegment,
    }));

    const totalHeight = segments.length * SEGMENT_HEIGHT;
    const visibleSegments = segments.slice(
      visibleRange.start,
      visibleRange.end,
    );

    return (
      <div
        ref={containerRef}
        className="h-full w-full"
        style={{ position: "relative", willChange: "transform" }}
      >
        <div style={{ height: `${totalHeight}px`, position: "relative" }}>
          {visibleRange.start > 0 && (
            <div
              style={{
                position: "absolute",
                top: 0,
                height: `${visibleRange.start * SEGMENT_HEIGHT}px`,
                width: "100%",
              }}
              aria-hidden="true"
            />
          )}
          {visibleSegments.map((segmentTime, index) => {
            const firstHalfMotionValue = getMotionSegmentValue(segmentTime);
            const secondHalfMotionValue = getMotionSegmentValue(
              segmentTime + segmentDuration / 2,
            );

            return (
              <div
                key={`${segmentTime}_${segmentDuration}`}
                style={{
                  position: "absolute",
                  top: `${(visibleRange.start + index) * SEGMENT_HEIGHT}px`,
                  height: `${SEGMENT_HEIGHT}px`,
                  width: "100%",
                }}
              >
                <MotionSegment
                  events={events}
                  firstHalfMotionValue={firstHalfMotionValue}
                  secondHalfMotionValue={secondHalfMotionValue}
                  segmentDuration={segmentDuration}
                  segmentTime={segmentTime}
                  timestampSpread={timestampSpread}
                  motionOnly={motionOnly}
                  showMinimap={showMinimap}
                  minimapStartTime={minimapStartTime}
                  minimapEndTime={minimapEndTime}
                  setHandlebarTime={setHandlebarTime}
                  scrollToSegment={scrollToSegment}
                  dense={dense}
                />
              </div>
            );
          })}
          {visibleRange.end < segments.length && (
            <div
              style={{
                position: "absolute",
                top: `${visibleRange.end * SEGMENT_HEIGHT}px`,
                height: `${(segments.length - visibleRange.end) * SEGMENT_HEIGHT}px`,
                width: "100%",
              }}
              aria-hidden="true"
            />
          )}
        </div>
      </div>
    );
  },
);
